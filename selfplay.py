import subprocess
import time
import os
from datetime import datetime

TRAIN_ENV = 'tictactoe'
CUDA_DEV = "2"
ROOT_DIR = f"/root/autodl-tmp"  # 写入的基本路径
BASE_MODEL = "Qwen2.5-1.5B-Instruct"
MODEL_DIR = f"{ROOT_DIR}/{TRAIN_ENV}/grpo"
START, END, STEP = 60, 200, 20
LOG_PATH = f"logs/{TRAIN_ENV}"

TIME = datetime.now().strftime("%m-%d-%H-%M")

# base config set
USE_GRPO="algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std actor_rollout_ref.actor.use_kl_loss=True"
USE_PPO="algorithm.adv_estimator=gae" # by default.
USE_BASE="algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1"



for i in range(START, END + 1, STEP):
    print(f"===== Round {i} =====")

    if i == 0:
        model_name = BASE_MODEL
        model_path = f"/{ROOT_DIR}/{model_name}"
    else:
        model_name = f"game_{i}"
        model_path = f"{MODEL_DIR}/{model_name}"

    # [1] 启动 vLLM serve
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CUDA_DEV
    vllm_cmd = [
        "vllm", "serve", model_path,
        "--port", "4040",
        "--max-model-len", "6000",
        "--host", "0.0.0.0",
        "--gpu-memory-utilization", "0.9"
    ]
    vllm_proc = subprocess.Popen(vllm_cmd, env=env, 
                                 stdout=open(f"{LOG_PATH}/vllm_{TIME}.log", "w"), 
                                 stderr=subprocess.STDOUT)
    print("[INFO] Waiting for vLLM to load...")
    time.sleep(60)  # 给 vllm 一点时间加载模型——顺利的话可以做到1分钟以内
    print('train models.')
    # [2] 运行训练
    # player_info = "[{model_name: '" + TRAIN_ENV + "/grpo/game_" + str(i) + "', port: '4040'}]"
    player_info = "{model_name: '" + TRAIN_ENV + "/grpo/game_{i}', port: '4040'}"
    # f"model_name: '{TRAIN_ENV}/grpo/game_{i}', port: '4040'}}]"
    print(f"[INFO] Set Env Player info {player_info}")
    # 开一个新的环境传入不使用wandb的环境变量
    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"
    train_cmd = [
        "python", "train.py",
        "--config-name", "_8_tictactoe",
        'system.CUDA_VISIBLE_DEVICES=\"0,1\"',
        f"model_path={ROOT_DIR}/{BASE_MODEL}",
        f'trainer.default_local_dir={ROOT_DIR}/{TRAIN_ENV}',
        f"trainer.total_training_steps={i + STEP}",
        "trainer.n_gpus_per_node=2",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=2",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.7",
        f"trainer.experiment_name={TRAIN_ENV}-grpo", 
        # f'custom_envs.TicTacToe.env_config.player_info="{player_info}"'
        f"custom_envs.TicTacToe.env_config.player_info=[{player_info}]"
    ]
    train_cmd += USE_GRPO.split()
    print(f"[INFO] Running training for {' '.join(train_cmd)}")
    subprocess.run(train_cmd, check=True, env=env,
                   stdout=open(f"{LOG_PATH}/grpo_{TIME}.log", "w"), 
                   stderr=subprocess.STDOUT)

    # [3] 模型转换
    # --backend fsdp --local_dir $ORI_PATH --target_dir $TAR_PATH --hf_model_path $HF_PATH
    convert_cmd = [
        "python", "verl/scripts/model_merger.py",
        "--backend fsdp",
        "--hf_model_path", f"{ROOT_DIR}/{BASE_MODEL}",
        "--local_dir", f"{ROOT_DIR}/{TRAIN_ENV}/global_step_{i + STEP}/actor",
        "--output", f"{MODEL_DIR}/game_{i + STEP}"
    ]
    subprocess.run(convert_cmd, check=True)

    # [4] 杀掉 vLLM serve
    print(f"[INFO] Killing vLLM serve (pid {vllm_proc.pid})")
    vllm_proc.terminate()
    vllm_proc.wait()
    # 如果存在 删除之前作为对手模型的参数文件
    MODEL_BEFORE = f"{ROOT_DIR}/{TRAIN_ENV}/global_step_{i}"
    if os.path.exists(MODEL_BEFORE):
        os.remove(MODEL_BEFORE)

    print(f"===== Round {i} Done =====\n")
