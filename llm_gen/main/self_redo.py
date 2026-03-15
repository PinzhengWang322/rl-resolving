import hydra
from omegaconf import DictConfig
from pprint import pprint
from transformers import AutoTokenizer
from tools.dset import VllmDataset
from vllm_gen.dp_infer import VLLMDataParallelInfer 
from tools.math_eval import evaluate_dataset
import os, json
from omegaconf import OmegaConf
import time

def evaluate_current_loop(cfg, dataset, tokenizer, name, tag, loop_id = 0):
    figs_dir = os.path.join(cfg.generation.save_dir, f'{tag}_figs')
    os.makedirs(figs_dir, exist_ok=True)
    fig_path = os.path.join(figs_dir, f'{name}.png')
    # ban_str = cfg.task.unsure_str if loop_id != cfg.task.max_loop else None
    ban_str = None
    record_str, results = evaluate_dataset(dataset, tokenizer, fig_path,
                                            ban_str=ban_str)
    with open(os.path.join(cfg.generation.save_dir, f'{tag}_{name}.log'), 'w') as f:
        f.write(record_str)
    metrics_dir = os.path.join(cfg.generation.save_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)  # 确保目录存在
    with open(os.path.join(metrics_dir, f'{name}.json'), 'w') as f:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        save_results = dict(
            config = cfg_dict,
            metrics = results
        )
        json.dump(save_results, f)


@hydra.main(config_path="config", config_name="math_solve", version_base=None)
def main(cfg: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.path)
    datasets = []
    for data_name in cfg.data_lst:
        print(f"solving datasets:{data_name}\n")
        vllm_dataset = VllmDataset(tokenizer, cfg.datas[data_name], cfg.data_format, 
                question_template_path = cfg.task.redo_prompt_path)
        datasets.append((data_name, vllm_dataset))
    
    engine = VLLMDataParallelInfer(model_path=cfg.model.path, 
                                   dp_size=cfg.model.dp_size, 
                                   tp_size=cfg.model.tp_size)
    
    for name, dataset in datasets:
        print(f"Inference dataset {name}:")
        engine.inference(dataset, 
                         temperature=cfg.generation.temperature,
                         top_p=cfg.generation.top_p,
                         max_tokens=cfg.generation.maxlen, 
                         gen_num=cfg.generation.gen_num)

        dataset.save(cfg.generation.save_dir, name)


if __name__ == "__main__":
    main()
