# OCT4LLM
One click tool for LLMs


## Installation

- Setup Docker

- Build Docker image

```bash
  docker build -t cuda_pytorch_docker .
```

    
- Run Docker container

```bash
  Docker run --gpus all -it --rm -v <your folder path>:/OCT4LLM cuda_pytorch_docker
```
