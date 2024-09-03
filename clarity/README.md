# Sharpness calculation and visualization tool

## Features

Simple tool of calculating sharpness

## Usage

#### Look up eligible options
```
$ python main.py -h                                                                            ok | yolo py | at 10:14:42 
usage: main.py [-h] [--hist] [--source SOURCE] [--save-csv] [--save-dir SAVE_DIR] [--size SIZE] [-v] [-f FPS]

Simple tool of calculating sharpness

optional arguments:
  -h, --help           show this help message and exit
  --hist               show the statistics as a histogram
  --source SOURCE      <image_path> or <video_path>
  --save-csv           save results to result.csv
  --save-dir SAVE_DIR  save the results to <save_dir>
  --size SIZE          save results to result.csv
  -v, --visualize      visualize clarity on the source, especially useful for video inputs
  -f FPS, --fps FPS    fps of visualization, only valid when used with <-v> and a video source
```

#### Run the tool
```
$ python main.py --source <your_source>
```

## Example