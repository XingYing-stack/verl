"""Ray-based demo for CE-Gate credit routing with VERL FSDP actor infrastructure."""

from __future__ import annotations

import argparse
import contextlib
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import ray
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from verl.single_controller.base.decorator import Dispatch, Execute, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.trainer.hooks.credit_router import CreditRouterConfig, GateCECreditRouter
from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.workers.fsdp_workers import ActorRolloutRefWorker


def _setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


LOGGER = _setup_logger("credit_demo")
WORKER_LOGGER = _setup_logger("credit_demo.worker")


def _log_gpu_stats(tag: str, logger: logging.Logger) -> None:
    if not torch.cuda.is_available():
        logger.info("GPU mem | tag=%s | device=cpu", tag)
        return
    try:
        torch.cuda.synchronize()
    except RuntimeError:
        pass
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    max_alloc = torch.cuda.max_memory_allocated()
    coeff = 1024 ** 3
    logger.info(
        "GPU mem | tag=%s | alloc=%.2f GiB | reserved=%.2f GiB | max_alloc=%.2f GiB | free=%.2f GiB | total=%.2f GiB",
        tag,
        allocated / coeff,
        reserved / coeff,
        max_alloc / coeff,
        free / coeff,
        total / coeff,
    )


def _resolve_config(cfg: DictConfig) -> DictConfig:
    """Return a resolved copy of the input config."""

    cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    OmegaConf.resolve(cfg_copy)
    return cfg_copy


def load_actor_config(args: argparse.Namespace) -> DictConfig:
    """Load the PPO trainer config and extract the actor block with overrides."""

    config_dir = Path(__file__).resolve().parents[1] / "verl" / "trainer" / "config"
    GlobalHydra.instance().clear()
    overrides = [
        f"actor_rollout_ref.model.enable_gradient_checkpointing={'true' if args.grad_ckpt else 'false'}",
        f"actor_rollout_ref.model.enable_activation_offload={'true' if args.activation_offload else 'false'}",
        "actor_rollout_ref.model.use_shm=false",
        "actor_rollout_ref.model.use_remove_padding=false",
        "actor_rollout_ref.model.use_fused_kernels=false",
        "actor_rollout_ref.rollout.name=hf",
        "actor_rollout_ref.rollout.n=1",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.prompt_length=16384",
        "actor_rollout_ref.rollout.response_length=1024",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false",
        "actor_rollout_ref.actor.strategy=fsdp",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={args.ppo_mini_batch_size}",
        "actor_rollout_ref.actor.ppo_micro_batch_size=null",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={args.ppo_micro_batch_size_per_gpu}",
        "actor_rollout_ref.actor.use_dynamic_bsz=false",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={'true' if args.param_offload else 'false'}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={'true' if args.optimizer_offload else 'false'}",
        f"actor_rollout_ref.actor.fsdp_config.offload_policy={'true' if args.offload_policy else 'false'}",
        "actor_rollout_ref.actor.fsdp_config.fsdp_size=-1",
        "actor_rollout_ref.actor.fsdp_config.forward_prefetch=false",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.optim.total_training_steps=1",
        "actor_rollout_ref.actor.optim.lr_warmup_steps=0",
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0",
        "actor_rollout_ref.actor.optim.weight_decay=0.0",
        "actor_rollout_ref.actor.optim.min_lr_ratio=0.0",
        "actor_rollout_ref.actor.optim.warmup_style=constant",
        "actor_rollout_ref.actor.optim.num_cycles=0.5",
    ]
    print(overrides)

    if args.model_path is not None:
        overrides.append(f"actor_rollout_ref.model.path={args.model_path}")

    with initialize_config_dir(version_base=None, config_dir=str(config_dir), job_name="credit_demo"):
        cfg = compose(config_name="ppo_trainer", overrides=overrides)

    cfg_resolved = _resolve_config(cfg)

    actor_cfg = cfg_resolved.actor_rollout_ref
    model_cfg = actor_cfg.model
    fsdp_cfg = actor_cfg.actor.fsdp_config

    LOGGER.info(
        "Resolved actor model flags | grad_ckpt=%s | activation_offload=%s | use_cache=%s | dtype=%s",
        model_cfg.get("enable_gradient_checkpointing", None),
        model_cfg.get("enable_activation_offload", None),
        model_cfg.get("use_cache", None),
        model_cfg.get("dtype", None),
    )
    LOGGER.info(
        "Resolved FSDP flags | param_offload=%s | optimizer_offload=%s | offload_policy=%s | forward_prefetch=%s",
        fsdp_cfg.get("param_offload", None),
        fsdp_cfg.get("optimizer_offload", None),
        fsdp_cfg.get("offload_policy", None),
        fsdp_cfg.get("forward_prefetch", None),
    )

    return actor_cfg


def build_router_config(args: argparse.Namespace) -> CreditRouterConfig:
    return CreditRouterConfig(
        enabled=True,
        rho_tool=args.rho_tool,
        length_norm=args.length_norm,
        temp=args.temp,
        chunk_logits=args.chunk_logits,
        adv_split_eta=args.adv_split_eta,
        route_mode=args.route_mode,
    )


def demo_prompt() -> str:
    import pickle

    with open('/share_data/data1/dialog_datasets/fsd_sharegpt/sft_data_0923_ASearcher_ZiqinPrompt_AllThought.pkl', 'rb') as f:
        data = pickle.load(f)
    return data[0][-50192:]

def base_worker_env() -> Dict[str, str]:
    return {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
    }


class CreditDemoWorker(ActorRolloutRefWorker):
    def __init__(self, config: DictConfig):
        super().__init__(config=config, role="actor")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, execute_mode=Execute.ALL)
    def compute_credit(
        self,
        chatml_samples: List[str],
        router_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not getattr(self, "_memlog_emitted", False):
            model_cfg = self.config.model if hasattr(self.config, "model") else {}
            actor_cfg = self.config.actor if hasattr(self.config, "actor") else {}
            actor_fsdp_cfg = actor_cfg.fsdp_config if hasattr(actor_cfg, "fsdp_config") else {}

            def _cfg_get(cfg, key):
                if hasattr(cfg, "get"):
                    try:
                        return cfg.get(key, None)
                    except Exception:
                        return None
                return None

            WORKER_LOGGER.info(
                "Worker config flags | grad_ckpt=%s | activation_offload=%s | param_offload=%s | optimizer_offload=%s | offload_policy=%s",
                _cfg_get(model_cfg, "enable_gradient_checkpointing"),
                _cfg_get(model_cfg, "enable_activation_offload"),
                _cfg_get(actor_fsdp_cfg, "param_offload"),
                _cfg_get(actor_fsdp_cfg, "optimizer_offload"),
                _cfg_get(actor_fsdp_cfg, "offload_policy"),
            )
            _log_gpu_stats("worker:init", WORKER_LOGGER)

            hf_model = getattr(self, "actor_module", None)
            if hf_model is not None:
                WORKER_LOGGER.info(
                    "HF model flags | gradient_checkpointing=%s | is_gradient_checkpointing=%s | use_cache=%s",
                    getattr(hf_model, "gradient_checkpointing", None),
                    getattr(hf_model, "is_gradient_checkpointing", None),
                    getattr(getattr(hf_model, "config", None), "use_cache", None),
                )
                expected_ckpt = _cfg_get(model_cfg, "enable_gradient_checkpointing")
                if expected_ckpt and not getattr(hf_model, "is_gradient_checkpointing", False):
                    WORKER_LOGGER.warning("Gradient checkpointing expected but not active on HF module")

            fsdp_module = getattr(self, "actor_module_fsdp", None)
            if fsdp_module is not None:
                cpu_offload = getattr(fsdp_module, "cpu_offload", None)
                mp_conf = getattr(fsdp_module, "_mixed_precision_config", None)
                mp_dtype = getattr(mp_conf, "param_dtype", None) if mp_conf is not None else None
                try:
                    sample_param = next(fsdp_module.parameters())
                    param_dtype = sample_param.dtype
                    param_device = sample_param.device
                except (StopIteration, AttributeError, TypeError):
                    param_dtype = None
                    param_device = None

                wrapped_layers = 0
                total_layers = 0
                for mod in fsdp_module.modules():
                    inner = getattr(mod, "_fsdp_wrapped_module", None)
                    if inner is None or isinstance(inner, torch.nn.Embedding):
                        continue
                    total_layers += 1
                    if hasattr(inner.forward, "__wrapped__"):
                        wrapped_layers += 1

                WORKER_LOGGER.info(
                    "FSDP runtime | type=%s | cpu_offload=%s | mp_param_dtype=%s | sample_param_dtype=%s | sample_param_device=%s",
                    type(fsdp_module).__name__,
                    cpu_offload,
                    mp_dtype,
                    param_dtype,
                    param_device,
                )
                if total_layers > 0:
                    WORKER_LOGGER.info(
                        "Activation offload instrumentation | wrapped_layers=%s/%s",
                        wrapped_layers,
                        total_layers,
                    )
                    expected_offload = _cfg_get(model_cfg, "enable_activation_offload")
                    if expected_offload and wrapped_layers == 0:
                        WORKER_LOGGER.warning("Activation offload requested but no wrapped transformer layers detected")
                WORKER_LOGGER.info(
                    "Worker offload flags | _is_offload_param=%s | _is_offload_optimizer=%s",
                    getattr(self, "_is_offload_param", None),
                    getattr(self, "_is_offload_optimizer", None),
                )
                expected_param_offload = _cfg_get(actor_fsdp_cfg, "param_offload")
                if expected_param_offload and not getattr(self, "_is_offload_param", False):
                    WORKER_LOGGER.warning("Param offload expected but worker flag is False")
                expected_optim_offload = _cfg_get(actor_fsdp_cfg, "optimizer_offload")
                if expected_optim_offload and not getattr(self, "_is_offload_optimizer", False):
                    WORKER_LOGGER.warning("Optimizer offload expected but worker flag is False")

            self._memlog_emitted = True

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if hasattr(self, "actor_module") and self.actor_module is not None:
            self.actor_module.config.use_cache = False
        # Always use the FSDP-wrapped model for credit to align with SFT path
        model_for_router = self.actor_module_fsdp
        router = GateCECreditRouter(CreditRouterConfig(**router_cfg), model=model_for_router, tokenizer=self.tokenizer)

        _log_gpu_stats("worker:pre_encode", WORKER_LOGGER)

        encoded = self.tokenizer(
            chatml_samples,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        batch = {k: v.to(device) for k, v in encoded.items()}
        group_adv = torch.ones(batch["input_ids"].size(0), device=device)

        WORKER_LOGGER.info(
            "compute_credit batch | shape=%s | attention_mask=%s | device=%s",
            tuple(batch["input_ids"].shape),
            tuple(batch["attention_mask"].shape) if "attention_mask" in batch else None,
            device,
        )
        _log_gpu_stats("worker:post_encode", WORKER_LOGGER)

        fsdp_ctx: contextlib.AbstractContextManager
        fsdp_ctx = contextlib.nullcontext()
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            # CE credit now runs directly on the sharded module, so avoid summon_full_params to let
            # FSDP manage sharding/offload itself (summon would reshard during forward).
            if isinstance(self.actor_module_fsdp, FSDP):
                pass
        except ImportError:
            pass

        # Ensure params on GPU when param_offload is enabled for FSDP v1
        # (safe to do before forward; do not offload after to avoid pointer churn).
        if self._is_offload_param:
            _log_gpu_stats("worker:before_load_fsdp", WORKER_LOGGER)
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
            _log_gpu_stats("worker:after_load_fsdp", WORKER_LOGGER)

        with fsdp_ctx:
            _log_gpu_stats("worker:before_router", WORKER_LOGGER)
            result = router(batch, group_adv)
            # Do not offload back here; leave it to FSDP after backward
            _log_gpu_stats("worker:after_router", WORKER_LOGGER)

        def _tensor_list(items: Iterable[torch.Tensor]) -> List[List[float]]:
            return [tensor.detach().cpu().tolist() for tensor in items]

        output = {
            "A_k": _tensor_list(result["A_k"]),
            "w_turn": _tensor_list(result["w_turn"]),
            "token_weight_shape": list(result["token_weight"].shape),
        }
        return output if self.rank == 0 else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ray-based credit routing demo")
    parser.add_argument(
        "--model-path",
        default="/share_data/data4/workhome/fanshengda/Qwen/Qwen3-8B",
        help="Path or HuggingFace id of the actor model (uses config default when omitted)",
    )
    parser.add_argument("--rho-tool", type=float, default=0.5, help="Weight for tool spans in credit aggregation")
    parser.add_argument("--length-norm", choices=["none", "len", "l2"], default="none")
    parser.add_argument("--temp", type=float, default=1.0, help="Softmax temperature for turn weighting")
    parser.add_argument("--chunk-logits", type=int, default=256, help="Vocabulary chunk size during CE accumulation")
    parser.add_argument("--adv-split-eta", type=float, default=0.7, help="Interpolation coefficient for GRPO split")
    parser.add_argument(
        "--route-mode",
        choices=["adv_split", "shaping"],
        default="adv_split",
        help="How to map turn weights to per-turn advantages",
    )
    parser.add_argument(
        "--grad-ckpt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable gradient checkpointing",
    )
    parser.add_argument(
        "--activation-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable activation offload",
    )
    parser.add_argument(
        "--param-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable FSDP parameter CPU offload",
    )
    parser.add_argument(
        "--optimizer-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable FSDP optimizer CPU offload",
    )
    parser.add_argument(
        "--offload-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable FSDP offload policy (FSDP2)",
    )
    parser.add_argument(
        "--model-dtype",
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Model dtype used during FSDP wrapping",
    )
    parser.add_argument(
        "--include-ray-dashboard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Launch Ray dashboard when initializing locally",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of FSDP ranks / GPUs to use",
    )
    parser.add_argument(
        "--ppo-mini-batch-size",
        type=int,
        default=1,
        help="Global PPO mini-batch size used for actor normalization",
    )
    parser.add_argument(
        "--ppo-micro-batch-size-per-gpu",
        type=int,
        default=1,
        help="Actor micro batch size per GPU",
    )
    return parser.parse_args()


def maybe_init_ray(enable_dashboard: bool) -> None:
    if ray.is_initialized():
        return
    runtime_env = {"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}}
    ray_kwargs: Dict[str, Any] = {
        "runtime_env": runtime_env,
        "ignore_reinit_error": True,
        "include_dashboard": enable_dashboard,
        "log_to_driver" : True,
        "logging_level" : logging.DEBUG
    }
    ray.init(**ray_kwargs)


def main() -> None:
    args = parse_args()
    LOGGER.info(
        "CLI args | grad_ckpt=%s | activation_offload=%s | param_offload=%s | optimizer_offload=%s | offload_policy=%s | model_dtype=%s | world_size=%s",
        args.grad_ckpt,
        args.activation_offload,
        args.param_offload,
        args.optimizer_offload,
        args.offload_policy,
        args.model_dtype,
        args.world_size,
    )
    actor_cfg = load_actor_config(args)
    router_cfg = build_router_config(args)

    maybe_init_ray(args.include_ray_dashboard)
    prompt = demo_prompt()

    assert args.world_size >= 1, "world_size must be >= 1"
    ray_cls = RayClassWithInitArgs(cls=ray.remote(CreditDemoWorker), config=actor_cfg)
    resource_pool = RayResourcePool(process_on_nodes=[args.world_size], use_gpu=True, name_prefix="credit_demo")
    wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls,
        device_name="cuda",
        worker_env=base_worker_env(),
    )
    wg.init_model()
    _log_gpu_stats("driver:post_init_model", LOGGER)
    raw_result = wg.compute_credit(chatml_samples=[prompt], router_cfg=asdict(router_cfg))
    _log_gpu_stats("driver:post_compute_credit", LOGGER)
    if isinstance(raw_result, list):
        result = next((item for item in raw_result if item is not None), raw_result[0])
    else:
        result = raw_result

    print("A_k per sample:", result["A_k"])
    print("w_turn per sample:", result["w_turn"])
    print("token_weight shape:", result["token_weight_shape"])
    ray.shutdown()


if __name__ == "__main__":
    main()
