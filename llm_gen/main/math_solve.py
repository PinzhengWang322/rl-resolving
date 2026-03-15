import hydra
from omegaconf import DictConfig
from pprint import pprint
from transformers import AutoTokenizer
from tools.dset import VllmDataset
from vllm_gen.dp_infer import VLLMDataParallelInfer 
from tools.math_eval import evaluate_dataset
import os, json
from omegaconf import OmegaConf

@hydra.main(config_path="config", config_name="math_solve", version_base=None)
def main(cfg: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.path)
    datasets = []
    for data_name in cfg.data_lst:
        vllm_dataset = VllmDataset(tokenizer, cfg.datas[data_name], cfg.data_format, 
                        question_template_path=cfg.task.solve_prompt_path)
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
