# Learning Sparse Graphon Mean Field Games

This repository is the official implementation of Learning Sparse Graphon Mean Field Games.

## Requirements

To install requirements:

```shell script
pip install -r requirements.txt
```

If needed, set PYTHONPATH to include the top-level folder, e.g.
```shell script
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Training

To train, run the following commands:

```shell script
python ./experiments/run.py --game=<game_name> --solver=<solver_name> --eval_solver=<solver_name> --simulator=<sim_name> --evaluator=<eval_name> --iterations=<num_iters> --eta=<temperature> --graphon=<graphon_name> --num_alphas=<num_discretization>
```

For example, you can run the following command to run 250 OMD iterations with eta of 1.0 on the Cyber Security problem with power law graphon.

```shell script
python ./experiments/run.py --game=Cyber-Graphon --solver=omd --simulator=exact --evaluator=exact --iterations=250 --eta=1 --graphon=power
```

For other options, please see the associated help or ./experiments/args_parser.py

All results will be saved in corresponding folders in ./results/ with full logs.

## Evaluation

To evaluate learned GMFEs on the N-agent environment as in the paper, run this command:

```shell script
python ./experiments/run_once_nagent_compare_sparse.py --num_players_point=<num_players> --game=<game_name> --graphon=<graphon_name> --fixed_alphas=0 --id=<seed> --beta=<beta>
```

See also the scripts in the experiments folder.
All results will be saved in the corresponding experiment folders in ./results/ with full logs. 

For more details, see the paper.
