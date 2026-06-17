import argparse
import json
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--root", default="/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results")
args = parser.parse_args()

root = Path(args.root)
manifest_path = root / "manifest.json"
bundle_path = root / "human_check_bundle"
bundle_path.mkdir(parents=True, exist_ok=True)
rows = []
combined_rows = []


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def parse_points(text):
    if not text:
        return []
    points = []
    for x, y in re.findall(r"\(\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\)", text):
        try:
            points.append([float(x), float(y)])
        except ValueError:
            pass
    return points


if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    for demo in manifest.get("selected", []):
        d = Path(demo["demo_dir"])
        geometry_file = d / "geometry_qwen2_5_vl.json"
        affordance_file = d / "affordance_robopoint.json"
        geometry = read_json(geometry_file)
        affordance = read_json(affordance_file)
        geometry_g_i = (geometry or {}).get("geometry_g_i") or {}
        affordance_a_i = (affordance or {}).get("affordance_a_i") or {}
        raw_robopoint_text = affordance_a_i.get("raw_robopoint_text")
        preferred_contact_points = affordance_a_i.get("preferred_contact_points") or parse_points(raw_robopoint_text)
        combined = {
            "demo_id": demo["id"],
            "task": demo.get("task"),
            "language_description": demo.get("language_description"),
            "demo_dir": str(d),
            "images": demo.get("review_images") or demo.get("absolute_images") or [],
            "geometry_model": (geometry or {}).get("model"),
            "geometry_g_i": geometry_g_i,
            "affordance_model": (affordance or {}).get("model"),
            "affordance_a_i": affordance_a_i,
            "source_files": {
                "geometry_qwen2_5_vl": str(geometry_file),
                "affordance_robopoint": str(affordance_file),
            },
        }
        demo_bundle_dir = bundle_path / demo["id"]
        demo_bundle_dir.mkdir(parents=True, exist_ok=True)
        (demo_bundle_dir / "combined_review.json").write_text(json.dumps(combined, indent=2))
        combined_rows.append(combined)
        rows.append({
            "demo_id": demo["id"],
            "task": demo.get("task"),
            "language_description": demo.get("language_description"),
            "demo_dir": str(d),
            "images": demo.get("review_images") or demo.get("absolute_images") or [],
            "geometry_file": str(geometry_file),
            "geometry_done": geometry_file.exists(),
            "affordance_file": str(affordance_file),
            "affordance_done": affordance_file.exists(),
            "geometry_key_features": geometry_g_i.get("key_features", []),
            "preferred_contact_points": preferred_contact_points,
            "grasp_affordance": affordance_a_i.get("grasp_affordance"),
            "contact_affordance": affordance_a_i.get("contact_affordance"),
            "motion_affordance": affordance_a_i.get("motion_affordance"),
            "required_contact_region": affordance_a_i.get("required_contact_region"),
            "combined_review_file": str(demo_bundle_dir / "combined_review.json"),
        })
(root / "review_index.json").write_text(json.dumps(rows, indent=2))
(root / "review_bundle.jsonl").write_text(
    "\n".join(json.dumps(row) for row in combined_rows) + "\n"
)
with (root / "review_index.md").open("w") as f:
    f.write("# Geometry/Affordance Human Review Index\n\n")
    f.write("Pilot outputs for human checking. Qwen2.5-VL is used only for geometry descriptions; RoboPoint is used only for affordance/contact/keypoint descriptions.\n\n")
    for r in rows:
        f.write(f"## {r['demo_id']}\n\n")
        f.write(f"- Task: {r['language_description']}\n")
        f.write(f"- Demo folder: `{r['demo_dir']}`\n")
        f.write(f"- Geometry, Qwen2.5-VL: `{r['geometry_file']}` ({'done' if r['geometry_done'] else 'missing'})\n")
        f.write(f"- Geometry key features: `{r['geometry_key_features']}`\n")
        f.write(f"- Affordance, RoboPoint: `{r['affordance_file']}` ({'done' if r['affordance_done'] else 'missing'})\n")
        f.write(f"- Affordance labels: grasp=`{r['grasp_affordance']}`, contact=`{r['contact_affordance']}`, motion=`{r['motion_affordance']}`, region=`{r['required_contact_region']}`\n")
        f.write(f"- Combined review: `{r['combined_review_file']}`\n")
        f.write(f"- RoboPoint preferred contact points: `{r['preferred_contact_points']}`\n")
        for img in r["images"]:
            f.write(f"- Image: `{img}`\n")
        f.write("\n")
print(root / "review_index.md")
