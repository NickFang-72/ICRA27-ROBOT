# Geometry/Affordance Human Review Index

Pilot outputs for human checking. Qwen2.5-VL is used only for geometry descriptions; RoboPoint is used only for affordance/contact/keypoint descriptions.

## turn_tap_episode161_0_71

- Task: turn left tap
- Demo folder: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/turn_tap_episode161_0_71`
- Geometry, Qwen2.5-VL: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/turn_tap_episode161_0_71/geometry_qwen2_5_vl.json` (done)
- Geometry key features: `['tap', 'handle', 'knob', 'rotational_axis', 'cylindrical']`
- Affordance, RoboPoint: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/turn_tap_episode161_0_71/affordance_robopoint.json` (done)
- Affordance labels: grasp=`knob_grasp`, contact=`rotate_part`, motion=`rotate`, region=`tap_handle_or_knob`
- Combined review: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/human_check_bundle/turn_tap_episode161_0_71/combined_review.json`
- RoboPoint preferred contact points: `[[0.358, 0.61], [0.359, 0.569], [0.328, 0.598], [0.389, 0.6], [0.33, 0.64]]`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/turn_tap_episode161_0_71/0.png`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/turn_tap_episode161_0_71/0.png`

## close_jar_episode161_0_65

- Task: close the red jar
- Demo folder: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/close_jar_episode161_0_65`
- Geometry, Qwen2.5-VL: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/close_jar_episode161_0_65/geometry_qwen2_5_vl.json` (done)
- Geometry key features: `['jar', 'round', 'cylindrical', 'hollow', 'lid', 'rim', 'top_opening']`
- Affordance, RoboPoint: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/close_jar_episode161_0_65/affordance_robopoint.json` (done)
- Affordance labels: grasp=`rim_grasp`, contact=`rotate_part`, motion=`twist`, region=`lid_or_rim`
- Combined review: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/human_check_bundle/close_jar_episode161_0_65/combined_review.json`
- RoboPoint preferred contact points: `[[0.389, 0.698], [0.358, 0.694], [0.391, 0.656], [0.359, 0.652]]`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/close_jar_episode161_0_65/0.png`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/close_jar_episode161_0_65/0.png`

## light_bulb_in_episode161_0_37

- Task: screw in the red light bulb
- Demo folder: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/light_bulb_in_episode161_0_37`
- Geometry, Qwen2.5-VL: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/light_bulb_in_episode161_0_37/geometry_qwen2_5_vl.json` (done)
- Geometry key features: `['light_bulb', 'round', 'bulb', 'threaded_base', 'socket', 'rotational_axis', 'alignment_sensitive']`
- Affordance, RoboPoint: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/light_bulb_in_episode161_0_37/affordance_robopoint.json` (done)
- Affordance labels: grasp=`body_grasp`, contact=`insert_object`, motion=`insert_then_twist`, region=`bulb_body_or_base`
- Combined review: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/human_check_bundle/light_bulb_in_episode161_0_37/combined_review.json`
- RoboPoint preferred contact points: `[[0.3, 0.7], [0.269, 0.681], [0.331, 0.685], [0.298, 0.658], [0.33, 0.644]]`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/light_bulb_in_episode161_0_37/0.png`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/light_bulb_in_episode161_0_37/0.png`

## slide_block_to_color_target_episode161_0_67

- Task: slide the block to green target
- Demo folder: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/slide_block_to_color_target_episode161_0_67`
- Geometry, Qwen2.5-VL: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/slide_block_to_color_target_episode161_0_67/geometry_qwen2_5_vl.json` (done)
- Geometry key features: `['block', 'rectangular', 'flat_faces', 'solid', 'target_region']`
- Affordance, RoboPoint: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/slide_block_to_color_target_episode161_0_67/affordance_robopoint.json` (done)
- Affordance labels: grasp=`body_grasp`, contact=`push_surface`, motion=`slide`, region=`block_side_or_top`
- Combined review: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/human_check_bundle/slide_block_to_color_target_episode161_0_67/combined_review.json`
- RoboPoint preferred contact points: `[[0.3, 0.69], [0.333, 0.683], [0.364, 0.69], [0.395, 0.692]]`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/slide_block_to_color_target_episode161_0_67/0.png`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/slide_block_to_color_target_episode161_0_67/0.png`

## sweep_to_dustpan_of_size_episode161_0_57

- Task: sweep dirt to the tall dustpan
- Demo folder: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/sweep_to_dustpan_of_size_episode161_0_57`
- Geometry, Qwen2.5-VL: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/sweep_to_dustpan_of_size_episode161_0_57/geometry_qwen2_5_vl.json` (done)
- Geometry key features: `['dustpan', 'open_container', 'thin_edge', 'flat_floor', 'target_region']`
- Affordance, RoboPoint: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/sweep_to_dustpan_of_size_episode161_0_57/affordance_robopoint.json` (done)
- Affordance labels: grasp=`handle_grasp`, contact=`push_surface`, motion=`push`, region=`sweeping_tool_or_object_side`
- Combined review: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/human_check_bundle/sweep_to_dustpan_of_size_episode161_0_57/combined_review.json`
- RoboPoint preferred contact points: `[[0.291, 0.69], [0.333, 0.69], [0.364, 0.692], [0.395, 0.694]]`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/sweep_to_dustpan_of_size_episode161_0_57/0.png`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/sweep_to_dustpan_of_size_episode161_0_57/0.png`

## push_buttons_episode161_0_51

- Task: push the maroon button
- Demo folder: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/push_buttons_episode161_0_51`
- Geometry, Qwen2.5-VL: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/push_buttons_episode161_0_51/geometry_qwen2_5_vl.json` (done)
- Geometry key features: `['button', 'round', 'small', 'raised_surface', 'flat_top']`
- Affordance, RoboPoint: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/push_buttons_episode161_0_51/affordance_robopoint.json` (done)
- Affordance labels: grasp=`none`, contact=`push_surface`, motion=`push`, region=`button_top`
- Combined review: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/human_check_bundle/push_buttons_episode161_0_51/combined_review.json`
- RoboPoint preferred contact points: `[[0.386, 0.698], [0.388, 0.656], [0.355, 0.69], [0.356, 0.648]]`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/push_buttons_episode161_0_51/0.png`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/push_buttons_episode161_0_51/0.png`

## put_groceries_in_cupboard_episode161_0_74

- Task: put the crackers in the cupboard
- Demo folder: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/put_groceries_in_cupboard_episode161_0_74`
- Geometry, Qwen2.5-VL: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/put_groceries_in_cupboard_episode161_0_74/geometry_qwen2_5_vl.json` (done)
- Geometry key features: `['cupboard', 'box_like', 'hollow', 'shelf', 'open_container']`
- Affordance, RoboPoint: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/put_groceries_in_cupboard_episode161_0_74/affordance_robopoint.json` (done)
- Affordance labels: grasp=`body_grasp`, contact=`lift_and_place`, motion=`lift_then_insert`, region=`object_body`
- Combined review: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/human_check_bundle/put_groceries_in_cupboard_episode161_0_74/combined_review.json`
- RoboPoint preferred contact points: `[[0.408, 0.7], [0.45, 0.721], [0.409, 0.75], [0.481, 0.719], [0.441, 0.762], [0.378, 0.717]]`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/put_groceries_in_cupboard_episode161_0_74/0.png`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/put_groceries_in_cupboard_episode161_0_74/0.png`

## put_money_in_safe_episode161_0_63

- Task: put the money away in the safe on the bottom shelf
- Demo folder: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/put_money_in_safe_episode161_0_63`
- Geometry, Qwen2.5-VL: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/put_money_in_safe_episode161_0_63/geometry_qwen2_5_vl.json` (done)
- Geometry key features: `['safe', 'rectangular', 'hollow', 'slot', 'front_opening']`
- Affordance, RoboPoint: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/put_money_in_safe_episode161_0_63/affordance_robopoint.json` (done)
- Affordance labels: grasp=`pinch`, contact=`insert_object`, motion=`insert`, region=`thin_object_edge`
- Combined review: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/human_check_bundle/put_money_in_safe_episode161_0_63/combined_review.json`
- RoboPoint preferred contact points: `[[0.389, 0.7], [0.43, 0.7], [0.358, 0.702], [0.461, 0.702]]`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/put_money_in_safe_episode161_0_63/0.png`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/put_money_in_safe_episode161_0_63/0.png`

## place_shape_in_shape_sorter_episode161_0_70

- Task: put the cube in the shape sorter
- Demo folder: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/place_shape_in_shape_sorter_episode161_0_70`
- Geometry, Qwen2.5-VL: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/place_shape_in_shape_sorter_episode161_0_70/geometry_qwen2_5_vl.json` (done)
- Geometry key features: `['shape_sorter', 'shape_profile', 'slot', 'hole', 'matching_geometry', 'alignment_sensitive']`
- Affordance, RoboPoint: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/place_shape_in_shape_sorter_episode161_0_70/affordance_robopoint.json` (done)
- Affordance labels: grasp=`edge_grasp`, contact=`insert_object`, motion=`insert`, region=`shape_body`
- Combined review: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/human_check_bundle/place_shape_in_shape_sorter_episode161_0_70/combined_review.json`
- RoboPoint preferred contact points: `[[0.3, 0.69], [0.333, 0.694], [0.364, 0.696], [0.395, 0.698]]`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/place_shape_in_shape_sorter_episode161_0_70/0.png`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/place_shape_in_shape_sorter_episode161_0_70/0.png`

## put_item_in_drawer_episode161_0_1

- Task: put the item in the bottom drawer
- Demo folder: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/put_item_in_drawer_episode161_0_1`
- Geometry, Qwen2.5-VL: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/put_item_in_drawer_episode161_0_1/geometry_qwen2_5_vl.json` (done)
- Geometry key features: `['drawer', 'rectangular', 'hollow', 'open_container', 'sliding_axis', 'handle']`
- Affordance, RoboPoint: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/put_item_in_drawer_episode161_0_1/affordance_robopoint.json` (done)
- Affordance labels: grasp=`body_grasp`, contact=`lift_and_place`, motion=`lift_then_place`, region=`object_body`
- Combined review: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/human_check_bundle/put_item_in_drawer_episode161_0_1/combined_review.json`
- RoboPoint preferred contact points: `[[0.286, 0.7], [0.333, 0.698], [0.369, 0.696], [0.314, 0.731], [0.35, 0.733]]`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/put_item_in_drawer_episode161_0_1/0.png`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/put_item_in_drawer_episode161_0_1/0.png`

## insert_onto_square_peg_episode161_0_63

- Task: put the ring on the red spoke
- Demo folder: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/insert_onto_square_peg_episode161_0_63`
- Geometry, Qwen2.5-VL: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/insert_onto_square_peg_episode161_0_63/geometry_qwen2_5_vl.json` (done)
- Geometry key features: `['peg', 'square_peg', 'hole', 'alignment_sensitive', 'insertable_part']`
- Affordance, RoboPoint: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/insert_onto_square_peg_episode161_0_63/affordance_robopoint.json` (done)
- Affordance labels: grasp=`edge_grasp`, contact=`insert_object`, motion=`insert`, region=`object_body`
- Combined review: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/human_check_bundle/insert_onto_square_peg_episode161_0_63/combined_review.json`
- RoboPoint preferred contact points: `[[0.381, 0.69], [0.383, 0.648], [0.38, 0.598], [0.38, 0.556], [0.383, 0.731]]`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/insert_onto_square_peg_episode161_0_63/0.png`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/insert_onto_square_peg_episode161_0_63/0.png`

## open_drawer_episode161_0_62

- Task: open the bottom drawer
- Demo folder: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/open_drawer_episode161_0_62`
- Geometry, Qwen2.5-VL: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/open_drawer_episode161_0_62/geometry_qwen2_5_vl.json` (done)
- Geometry key features: `['drawer', 'rectangular', 'hollow', 'handle', 'sliding_axis', 'front_face']`
- Affordance, RoboPoint: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/open_drawer_episode161_0_62/affordance_robopoint.json` (done)
- Affordance labels: grasp=`handle_grasp`, contact=`pull_handle`, motion=`pull`, region=`handle`
- Combined review: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/human_check_bundle/open_drawer_episode161_0_62/combined_review.json`
- RoboPoint preferred contact points: `[[0.286, 0.798], [0.333, 0.798], [0.255, 0.796], [0.364, 0.798]]`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/open_drawer_episode161_0_62/0.png`
- Image: `/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results/demos/open_drawer_episode161_0_62/0.png`
