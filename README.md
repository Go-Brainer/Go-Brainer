# Go-Brainer
The Go-Brainer project is an assignment for CSUF Fall 2019 CPSC 481: Artificial Intelligence with Dr. William McCarthy

## Project Description
Use Deep Learning techniques from lectures and the book *Deep Learning and the Game of Go* by Max Pumperla and Kevin Ferguson to develop a artificial intelligence bot that can play the board game Go.

## Participants:
Nick Bernstein  skolyr@csu.fullerton.edu  
Alex Vidal      avidal@csu.fullerton.edu

## Usage
C:\PATH\>python Go-Brainer.py -h
usage: Go-Brainer [-h] [-s SIZE] [-k KOMI] {h,r,d,a,m} {h,r,d,a,m}

An AI Go Bot

positional arguments:
  {h,r,d,a,m}           Choose Player Agent for black: h:human, r:random,
                        d:depth-pruned, a:alpha-beta, m:Mote-Carlo-Tree_Search
  {h,r,d,a,m}           Choose Player Agent for white: h:human, r:random,
                        d:depth-pruned, a:alpha-beta, m:Mote-Carlo-Tree_Search

optional arguments:
  -h, --help            show this help message and exit
  -s SIZE, --size SIZE  Board size is n by n. Must be between 5 and 19.
  -k KOMI, --komi KOMI  Sets komi. Must be between 0 and 10.

## Checkboxes

**To run MCTS against itself**
  ?>python Go-Brainer -m -m
**To generate games**
usage: generate_mcts_games.py [-h] [--board-size BOARD_SIZE] [--rounds ROUNDS]
                              [--temperature TEMPERATURE]
                              [--max-moves MAX_MOVES] [--num-games NUM_GAMES]
                              [--board-out BOARD_OUT] [--move-out MOVE_OUT]

optional arguments:
  -h, --help            show this help message and exit
  --board-size BOARD_SIZE, -b BOARD_SIZE
  --rounds ROUNDS, -r ROUNDS
  --temperature TEMPERATURE, -t TEMPERATURE
  --max-moves MAX_MOVES, -m MAX_MOVES
                        Max moves per game.
  --num-games NUM_GAMES, -n NUM_GAMES
  --board-out BOARD_OUT
  --move-out MOVE_OUT

**CNN from 6.24-6.26**
./cnn_training./mcts_cnn_go_40k_output.txt

**KGS separate download script**
python ./data/index_processor.py

**KGS download with training generators**
python ./train_generator


