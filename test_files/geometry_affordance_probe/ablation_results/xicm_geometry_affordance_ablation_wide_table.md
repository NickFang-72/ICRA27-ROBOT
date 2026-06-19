# X-ICM Geometry/Affordance Ablation Results

Scores are numeric final success percentages from 25 evaluation episodes per task. Accuracy is score / 100 in the CSV.

## Completion

- original_xicm: 23/23 task scores
- geometry: 23/23 task scores
- affordance: 0/23 task scores
- geometry_affordance: 0/23 task scores

## Summary

| task | original_xicm_score | geometry_score | affordance_score | geometry_affordance_score | best_run | best_score | geometry_delta_vs_original | affordance_delta_vs_original | geometry_affordance_delta_vs_original |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MEAN_ALL | 22.09 | 20.17 |  |  | original_xicm | 22.09 | -1.91 |  |  |
| MEAN_LEVEL_1 | 28 | 24.31 |  |  | original_xicm | 28 | -3.69 |  |  |
| MEAN_LEVEL_2 | 14.4 | 14.8 |  |  | geometry | 14.8 | 0.4 |  |  |

## Task Results

| task | original_xicm_score | geometry_score | affordance_score | geometry_affordance_score | best_run | geometry_delta_vs_original | affordance_delta_vs_original | geometry_affordance_delta_vs_original |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| put_toilet_roll_on_stand | 0 | 0 |  |  | original_xicm | 0 |  |  |
| put_knife_on_chopping_board | 20 | 8 |  |  | original_xicm | -12 |  |  |
| close_fridge | 20 | 4 |  |  | original_xicm | -16 |  |  |
| close_microwave | 48 | 36 |  |  | original_xicm | -12 |  |  |
| close_laptop_lid | 40 | 32 |  |  | original_xicm | -8 |  |  |
| phone_on_base | 56 | 20 |  |  | original_xicm | -36 |  |  |
| toilet_seat_down | 60 | 64 |  |  | geometry | 4 |  |  |
| lamp_off | 60 | 84 |  |  | geometry | 24 |  |  |
| lamp_on | 40 | 52 |  |  | geometry | 12 |  |  |
| put_books_on_bookshelf | 0 | 0 |  |  | original_xicm | 0 |  |  |
| put_umbrella_in_umbrella_stand | 0 | 4 |  |  | geometry | 4 |  |  |
| open_grill | 4 | 12 |  |  | geometry | 8 |  |  |
| put_rubbish_in_bin | 16 | 0 |  |  | original_xicm | -16 |  |  |
| take_usb_out_of_computer | 96 | 88 |  |  | original_xicm | -8 |  |  |
| take_lid_off_saucepan | 16 | 0 |  |  | original_xicm | -16 |  |  |
| take_plate_off_colored_dish_rack | 0 | 0 |  |  | original_xicm | 0 |  |  |
| basketball_in_hoop | 4 | 0 |  |  | original_xicm | -4 |  |  |
| scoop_with_spatula | 0 | 0 |  |  | original_xicm | 0 |  |  |
| straighten_rope | 8 | 8 |  |  | original_xicm | 0 |  |  |
| turn_oven_on | 16 | 40 |  |  | geometry | 24 |  |  |
| beat_the_buzz | 0 | 0 |  |  | original_xicm | 0 |  |  |
| water_plants | 0 | 12 |  |  | geometry | 12 |  |  |
| unplug_charger | 4 | 0 |  |  | original_xicm | -4 |  |  |
