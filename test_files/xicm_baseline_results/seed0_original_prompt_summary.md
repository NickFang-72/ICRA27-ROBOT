# X-ICM Baseline on AGNOSTOS Unseen Tasks

- Prompt: original X-ICM paper prompt, no geometry or affordance descriptions.
- Model: Qwen2.5-7B-Instruct via the X-ICM pipeline.
- Seed: 0
- Episodes per task: 25
- In-context demos: 18
- Retrieval mode: lang_vis.out
- Seen task folders linked: 18
- Unseen task folders linked: 23
- Result CSV files: 23
- Completed task scores: 23
- Mean success score: 22.087

## Scores

| Task | Score |
| --- | ---: |
| put_toilet_roll_on_stand | 0 |
| put_knife_on_chopping_board | 20 |
| close_fridge | 20 |
| close_microwave | 48 |
| close_laptop_lid | 40 |
| phone_on_base | 56 |
| toilet_seat_down | 60 |
| lamp_off | 60 |
| lamp_on | 40 |
| put_books_on_bookshelf | 0 |
| put_umbrella_in_umbrella_stand | 0 |
| open_grill | 4 |
| put_rubbish_in_bin | 16 |
| take_usb_out_of_computer | 96 |
| take_lid_off_saucepan | 16 |
| take_plate_off_colored_dish_rack | 0 |
| basketball_in_hoop | 4 |
| scoop_with_spatula | 0 |
| straighten_rope | 8 |
| turn_oven_on | 16 |
| beat_the_buzz | 0 |
| water_plants | 0 |
| unplug_charger | 4 |

## Artifacts

- Main log: /data/yf23/projects/ICRA27-ROBOT/X-ICM/logs/baseline_xicm_original_prompt/run_20260618_210948.log
- Result directory: /data/yf23/projects/ICRA27-ROBOT/X-ICM/logs/XICM_Cross.ZS_Ranking.lang_vis.out_Qwen2.5.7B.instruct_icl.18_test
- Score CSV: /data/yf23/projects/ICRA27-ROBOT/X-ICM/logs/baseline_xicm_original_prompt/seed0_original_prompt_scores.csv
