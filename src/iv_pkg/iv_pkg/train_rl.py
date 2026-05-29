# train_rl.py
# PPO training for (lookahead, steering_gain) — ROS2-friendly script

import os
import math
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# adjust this import to your package structure if needed
from iv_pkg.pure_env import F1TenthEnv

def get_base_env(vec_env):
    env = vec_env
    if isinstance(env, VecNormalize):
        env = env.venv            # -> DummyVecEnv
    base = env.envs[0]            # -> Monitor
    if isinstance(base, Monitor):
        base = base.env           # -> F1TenthEnv
    return base


def make_env(seed: int = 0):
    def _thunk():
        env = F1TenthEnv()
        env = Monitor(env)  # episode stats for TB
        return env
    set_random_seed(seed)
    return _thunk


def build_dirs(outdir: str):
    logs = os.path.join(outdir, "ppo_logs")
    best = os.path.join(outdir, "best_model")
    ckpt = os.path.join(outdir, "checkpoints")
    os.makedirs(logs, exist_ok=True)
    os.makedirs(best, exist_ok=True)
    os.makedirs(ckpt, exist_ok=True)
    return logs, best, ckpt


def cosine_lr(frac: float, base: float = 2.4e-4):
    # Optional: smoother than linear; switch to this by passing --lr-schedule cosine
    return base * 0.5 * (1.0 + math.cos(math.pi * (1.0 - frac)))


def main():
    parser = argparse.ArgumentParser(description="Train PPO for (L_d, gain)")
    parser.add_argument("--outdir", type=str, default=".",
                        help="Output directory for logs/checkpoints/models")
    parser.add_argument("--timesteps", type=int, default=1_200_000,
                        help="Total training timesteps")
    parser.add_argument("--eval-freq", type=int, default=5_000,
                        help="Eval frequency (steps)")
    parser.add_argument("--n-eval-episodes", type=int, default=8,
                        help="Episodes per evaluation")
    parser.add_argument("--ckpt-freq", type=int, default=25_000,
                        help="Checkpoint save frequency (steps)")
    parser.add_argument("--seed", type=int, default=0, help="Training seed")
    parser.add_argument("--lr", type=float, default=2.4e-4, help="Base learning rate")
    parser.add_argument("--lr-schedule", choices=["linear", "cosine"], default="linear",
                        help="Learning rate schedule")
    args = parser.parse_args()

    LOGS_DIR, BEST_DIR, CKPT_DIR = build_dirs(args.outdir)

    # --- VecEnvs ---
    train_env = DummyVecEnv([make_env(args.seed)])
    train_env = VecNormalize(
        train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0
    )

    # eval env shares stats but does not update them
    eval_env = DummyVecEnv([make_env(999)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, training=False)
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms

    # --- Callbacks ---
    stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10, min_evals=5, verbose=1
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=BEST_DIR,
        log_path=LOGS_DIR,
        eval_freq=args.eval_freq,
        deterministic=True,
        n_eval_episodes=args.n_eval_episodes,
        #callback_after_eval=stop_cb,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=args.ckpt_freq, save_path=CKPT_DIR, name_prefix="ppo"
    )

    # --- Model ---
    if args.lr_schedule == "linear":
        lr_fn = lambda frac: args.lr * frac
    else:
        lr_fn = lambda frac: cosine_lr(frac, base=args.lr)

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=lr_fn,
        n_steps=4096,
        batch_size=256,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.98,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.6,
        max_grad_norm=0.7,
        target_kl=0.015,
        tensorboard_log=LOGS_DIR,
        seed=args.seed,
    )

    # (Optional) If your env implements attach_logger(), you could do:
    # train_env.envs[0].env.attach_logger(model.logger)
    # eval_env.envs[0].env.attach_logger(model.logger)

    
    get_base_env(train_env).attach_logger(model.logger)
    get_base_env(eval_env).attach_logger(model.logger)

    print("[PPO] Starting training…")

    model.learn(total_timesteps=args.timesteps, callback=[eval_cb, ckpt_cb])

    # --- Save ---
    model_path = os.path.join(args.outdir, "ppo_lookahead_gain_model")
    norm_path = os.path.join(args.outdir, "vecnorm.pkl")
    model.save(model_path)
    train_env.save(norm_path)

    #train_env.envs[0].env.attach_logger(model.logger)
    #eval_env.envs[0].env.attach_logger(model.logger)

    print(f"[PPO] Saved model to: {model_path}.zip")
    print(f"[PPO] Saved VecNormalize stats to: {norm_path}")


if __name__ == "__main__":
    main()
