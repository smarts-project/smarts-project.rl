# Creating Scenarios
<!-- 
<img align="right" width="400" height="400" src="research/edmonton/media/intersections/stopwatcher.gif"/> -->

## Summary of Contents:
- **analysis/**: Contains scripts used for the different types of analysis.
- **common/**: 
  - `distributions.py`: A script for defining different traffic patterns.
  - `social_vehicle_definitions.py`: Contains different social behaviors.
  - `visualization.py`: A tool for visualizing scenarios.
- **pool/**: Contains different maps with various...
  - **analysis_pool/**: Contains different maps for analysis with various...
    - number of lanes (2, 3, 4, 5, and 6),
    - max speeds (50kmh, 70kmh, 100kmh), and
    - shapes (c: cross-intersection, t: t-intersection).
  - **experiment_pool/**: Contains different maps for experiments with various...
    - number of lanes (2, 3, 4, 5, and 6),
    - max speeds (50kmh, 70kmh, 100kmh), and
    - shapes (c: cross-intersection, t: t-intersection).
  > Note that the analysis pool and experiment pool contain the exact same maps except that the analysis pool has a side road on its maps that is used for analysis.
- **task1/**: 
  - `config.yaml`: Defines the ego mission and traffic patterns for Task 1 (generalize over intersections by training on t-intersections, testing on cross-intersections).
- **task2/**:
  - `config.yaml`: Defines the ego mission and traffic patterns for Task 2 (generalize over traffic density by training on high or low traffic and testing on low or high traffic, respectively).
  
## Conducting Analysis:
1. Generate maps:
    ```sh
    $ scl scenario build-all ultra/scenarios/pool/analysis_pool/
    ```
2. Define or modify the task in your task's config.yaml:
    ```yaml
    ego_missions: 
      # can define multiple missions
      # (example: left turn from south to west)
      start: south-SN
      end:   west-EW
    levels:
      easy:
        train:
          total: 800
          intersection_types:
            2lane_t: # train on t-intersections
              percent: 0.5
              specs: [[50kmh,low-density,0.33],[70kmh,low-density,0.33],[100kmh,low-density,0.33]]
            3lane_t:
              percent: 0.5
              specs: [[50kmh,low-density,0.33],[70kmh,low-density,0.33],[100kmh,low-density,0.33]]
        test:
          total: 200
          intersection_types:
            2lane_c: # test on cross-intersections
              percent: 0.5
              specs:  [[50kmh,low-density,0.33],[70kmh,low-density,0.33],[100kmh,low-density,0.33]]
            3lane_c:
              percent: 0.5
              specs: [50kmh,low-density,0.33],[70kmh,low-density,0.33],[100kmh,low-density,0.33]]
      hard:
        # same pattern with different densities
    ``` 
3. Generate scenarios for training:
    ```sh
    # Remove existing easy scenarios.
    $ rm -rf ultra/scenarios/task1/t*_task1_easy_*

    # Generate the analysis scenarios from the analysis maps in the analysis pool.
    $ python ultra/scenarios/interface.py generate --task 1 --level easy --pool-dir ultra/scenarios/pool/analysis_pool/
    ```
    Available parameters:
    - `--task`: Selects the task to generate (based on the task's config.yaml).
    - `--level`: Selects the level of the task from the task's config.yaml.
    - `--stopwatcher`: Include a stopwatcher (a vehicle that records the number of steps taken to do the left turn) in the scenarios.
    - `--save-dir`: The directory for saving the completed scenarios (completed scenarios will be put in the task directory if not specified).
    - `--root-dir`: The directory containing the task to be created (default is ultra/scenarios).
    - `--pool-dir`: The directory containing the maps for the scenarios (default is ultra/scenarios/pool/experiment_pool).
    - `--seed`: The seed used to generate the random seeds for the scenarios. If set, random seeds for the scenarios will be randomly sampled without replacement from the range `[0, 2 ** 31)`. By default, this argument is `None`, producing scenarios with seeds `0, 1, 2, ...`.
    > Each scenario generates a `metadata.json` file to show the routes, number of vehicles, and types of vehicles that are used in that scenario.
4. Do the analysis:
   - **Analyze scenarios**:
      ```sh
      $ python ultra/scenarios/interface.py analyze --scenarios ultra/scenarios/task1/<analysis_scenarios_pattern> --max-steps 6000 --video 100 --output <path_to_output_directory>
      ```
      This will...
      - run the scenario for 6000 steps and records all vehicle states for analysis,
      - save the analysis.pkl in the output directory, and
      - if `--video 100` is passed, every 100 scenarios, the analysis will save a GIF of that scenario.  
   - **Analyze behaviors**:
      ```sh
      $ python ultra/scenarios/interface.py behavior --scenarios ultra/scenarios/task1/<analysis_scenarios_pattern> --video 100 --output <path_to_output_directory>
      ```
    > NOTE: Video saving is very time consuming, avoid using it for long runs.
