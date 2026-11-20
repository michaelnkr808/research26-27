# PI follow-ups after the 2026-11-19 meeting: audio in WNM-3D's simulator, the two linked papers, and gaps in audio-VLA studies (2026-11-19)

Direction: hear-lab. Requested by the planning lead on 2026-11-19 as the follow-ups named in the PI's feedback (`LOG.md`, entry "2026-11-19 ... PI feedback"): "if you can use WNM-3D's own simulator for simulating audio-visual training data, then we will not use SoundSpaces 2.0"; two linked papers (arXiv 2610.21588 and Ryu et al., CVPR 2026); and "pursue research in a direction that addresses aspects not covered in existing audio-VLA studies (e.g., long-form processing)".

Written by three literature-session workers (one per part) and assembled by the literature lead; each part keeps its own method note. Labels: `verified` = read in the named source this session; `recalled` = from an earlier survey file or memory, not reopened; `speculative` = this file's own reading. Downloads were kept in a scratch directory outside the repository, treated as untrusted, and never executed. Sources that refused access were not worked around. Numbers are copied from sources; none were reproduced.

## Part A: can WNM-3D's own simulator make audio-visual training data?

Date: 2026-11-19 (session date). Direction: hear-lab (spatial audio for world action models). Requested by the planning lead after the PI's 2026-11-19 message: "And if you can use WNM-3D's own simulator for simulating audio-visual training data, then we will not use SoundSpaces 2.0" (hear-lab/LOG.md, entry "2026-11-19 ... PI feedback"). Extends Part 2 of `hear-lab/literature/survey/2026-11-18-novelty-and-simulator-gap.md` (branch origin/hear-lab-novelty-simulator-gap), which read only GN-Bench documentation and file lists.

Labels: `verified` = read this session in the named file, commit or page; `recalled` = from memory or an earlier survey file, not reopened; `speculative` = my inference. Repositories were read as tarballs at pinned commits (GitHub API) and grepped locally; Hugging Face datasets through the public API, the public dataset page, and the datasets-server first-rows endpoint. No gated payload was downloaded. One 256 KiB byte range of one ungated collision-mesh file was fetched to see its format (binary USD). Downloads are in the scratchpad `downloads/A` folder, untrusted, nothing executed. No hour or cost estimates are given. Nothing was run, so no claim here says that a pipeline "works".

### Answer

**Verdict: possible with named extra work, not as released.** GN-Bench renders RGB, depth and a top-down view from Gaussian splats and has no audio path at all. But InteriorGS ships a floor-plan file (rooms, walls with heights and thicknesses, door and window openings) and per-object 3D boxes with semantic labels, and the InteriorGS authors separately release per-object triangle collision meshes for all 1,000 scenes. That is enough geometry to drive a geometric acoustic engine. The most direct engine is RLR-Audio-Propagation (the engine inside SoundSpaces 2.0), which has a standalone C API that takes boxes or triangle meshes plus label-matched materials and returns mono, binaural or ambisonic impulse responses. Materials would be guessed from object labels (InteriorGS has no acoustic material data). I found nobody who has rendered geometric room acoustics on InteriorGS, GN-Bench or any Gaussian-splat navigation simulator.

What this means for the PI's condition (`speculative`): "use WNM-3D's own simulator" is achievable for the visuals, scenes and episodes, but the audio itself would still come from an external acoustic engine. If that engine is RLR, the audio physics is the same as SoundSpaces 2.0's; what changes is the scenes (InteriorGS instead of Matterport3D/HM3D), the episodes (GN-Matrix) and the visual renderer (Gaussian splats). The user should check with the PI whether that counts as "not using SoundSpaces 2.0".

#### 1. Does anything in GN0 / GN-Bench / WNM-3D render audio? No (`verified`)

- GN-Bench-Tools (the simulator, a git submodule of GN0, pinned at commit `0ce83677532591c582b0c8a2879ab431a1efff18`, which is also its current master) renders only through `gsplat`: `GN_Bench/render/gsplat_renderer.py` imports `from gsplat.rendering import rasterization` and returns RGB, or RGB plus expected depth (`render_mode = "RGB+ED" if render_depth else "RGB"`), plus an orthographic bird's-eye view. The simulator (`GN_Bench/sims/GN_Bench_simulator/GN_Bench_simulator.py`) registers exactly two sensors, `GNBenchSimRGBSensor` and `GNBenchSimDepthSensor`.
- Its `requirements.txt` lists numpy, numpy-quaternion, scipy, numba, h5py, plyfile, opencv-python, pillow, imageio, imageio-ffmpeg, gymnasium, yacs, tqdm, joblib, attrs. No audio or acoustics library. `setup.py` adds nothing beyond these.
- Case-insensitive grep for `audio|sound|acoustic|\brir\b|\.wav|binaural|ambison|reverb|impulse` over GN-Bench-Tools, GN0 (`c92778118542bf1dd7d437855d06e0d822e8361d`), WNM-3D (`7fe2d5f93a8dc4e2a5a2ec6fe174746928ecdcbb`) and GN0-VLN-CE (`ac4ebedb4ec88151656c3caa7cbcb93095497b6a`): zero hits in GN-Bench-Tools. The only hits elsewhere are `torchaudio` pins in the install guides, `"has_audio": False` in the LeRobot video metadata (`gn0/collection/dagger.py` line 750, `scripts/data/convert_interiorgs_to_lerobot.py` line 710), and `AUDIO_SAMPLING_RATE = 16000` with `setattr(processor, "use_audio_in_video", False)` in the GN-BAE inference file (a setting on the vision-language processor, turned off).
- Papers: the GN0 paper (arXiv 2606.03682 v1, HTML) and the WNM-3D paper (arXiv 2608.07267 v3, HTML) contain zero matches for audio, sound, acoustic or binaural. GN0 Sec. 4.2.1 lists the observation space as egocentric RGB, egocentric depth and top-down BEV.
- How a scene is loaded (`GN_Bench_simulator.py`, `reconfigure`, lines 594 to 702): `occupancy.json` and `occupancy.png` (2D grid for A* path planning and collision), `3dgs_compressed.ply` (the splats), the episode's trajectory payload, and optional human-avatar sequences. Agent height is fixed: `_load_initial_state` sets position z to `1.3` (line 780). World-to-splat coordinates go through a per-episode affine fit (`derive_affine_transform`, `transform_from_world_to_GS`). The simulator never opens `labels.json` or `structure.json`.

#### 2. What InteriorGS and GN-Matrix contain (`verified`)

InteriorGS (https://huggingface.co/datasets/spatialverse/InteriorGS, revision `334dfeea4e0241033b4e5de97c01bc7c9c080530`, last modified 2026-07-27; gated "auto", license "InteriorGS Terms of Use"; the card's dataset structure section was read on the public dataset page, the file list from the API):
- 1,000 scene folders; each has exactly five files: `3dgs_compressed.ply`, `labels.json`, `occupancy.json`, `occupancy.png`, `structure.json` (API file list, counted). Plus a top-level `visualize_meta.py`. No triangle meshes, no textures, no material files.
- Card: "Each scene is semantically annotated at the object level, with more than 554,000 object instances across 755 categories, and each object is paired with a 3D bounding box." Splats: "3D Gaussian parameters including position, covariance, opacity, and spherical harmonics coefficients", coordinate system "XYZ = (Right, Back, Up)", units meters.
- `labels.json`: "3D oriented bounding boxes defined by 8 corner vertices"; "Per-object semantic class labels with instance IDs".
- `occupancy.png/json`: "Top-down view of navigable floor space", values 255 free, 0 occupied, 127 unknown.
- `structure.json` ("Floorplan data", "right-handed, z up, and unit in meter"): `rooms` with `profile` ("2D polygon in xy plane that defines room outline") and `room_type` (e.g. "Living and Dining Room"); `walls` with `location` ("line segment in xy plane"), `thickness`, `height` ("height in z axis from ground(z=0)"), and `holes` ("openings on the walls", with a 3D `profile`, `thickness`, and `type` "WINDOW or DOOR"); `ins` with `label`, `position` ("instance bounding box center in world frame") and `size`.
- Card: the scenes were made "by photorealistically rendering handcrafted 3D environments, then reconstructing splatting-based representations". So artist meshes exist upstream, but InteriorGS itself does not ship them.

GN-Matrix (https://huggingface.co/datasets/TeleEmbodied/GN-Matrix, revision `50a68253b5436f0595779cd3f91baf65228a0238`, last modified 2026-07-15, not gated, CC BY-NC 4.0; card is only the license line):
- Files: `InteriorGS/InteriorGS_{all,train_seen,test_seen,test_unseen}.parquet`.
- Columns (datasets-server first-rows): `split, trajectory_id, scene, label_ins_id, label_label, label_goal_pixel_x/y, start_world_x/y, start_facing_unit_dx/dy, start_facing_heading_degrees, goal_world_x/y, grounded_instruction, instruction, path_raster_world_x/y, path_raster_pixel_x/y`. First row example: scene `0003_839989`, goal `label_label` "wardrobe", instruction "Go to the wardrobe in the dining room". The loader (`gn0/benchmark/dataset.py`) reads exactly these columns.
- So every episode names its goal object by instance id and label, which can be looked up in that scene's `labels.json` (`speculative` that the ids match; the column name and the card wording suggest it, the gated file was not opened).

Outside GN0, from the InteriorGS authors (SAGE-3D, arXiv 2510.21307 v2, ICLR 2026; repository https://github.com/Galery23/SAGE-3D_Official, pushed 2026-07-27):
- Paper Sec. 2.3: "we take the artist-created triangle meshes of each object and apply CoACD (Wei et al., 2022) for convex decomposition, yielding per-object collision bodies". The released environment runs in Isaac Sim 5.0+ and "provides synchronized RGB, depth, semantic segmentation, poses, and contact events". No audio in the paper text searched.
- Dataset https://huggingface.co/datasets/spatialverse/SAGE-3D_Collision_Mesh (revision `3ba75cc7887b62bf84211d5db08adfa64d691597`, not gated, CC BY-NC 4.0): 1,000 files `Collision_Mesh/<id>/<id>_collision.usd`; the ids match all 1,000 InteriorGS scene-folder suffixes (counted from the two API file lists). Card: "554,000+ collision bodies corresponding to object instances across 1000 indoor scenes", "Convex hull decomposition". One file (scene 839920) is 46,675,012 bytes and starts with the `PXR-USDC` binary header. I could not list its contents (no USD library installed, and pip installs are not allowed in this setup), so whether walls, floors and ceilings are included is not verified. The paper's overview says "scene and object collision bodies are generated via convex hull decomposition", which suggests the room shell is included (`speculative`).
- Also released: `spatialverse/SAGE-3D_InteriorGS_usdz` (splats as USDZ for Isaac Sim, not gated).

#### 3. Which of these could feed an acoustic simulator (`speculative` unless marked)

- Room shell: `structure.json` walls (segment, thickness, height) and room polygons can be extruded into floor, ceiling and wall triangles, with door holes left open. This is the geometry that dominates reverberation and is what tells sound from another room arrives through a doorway. Door state (open or closed) is not in the card's key list.
- Furniture: either the 3D boxes (`labels.json` or `structure.json` `ins`) as box obstacles, or the SAGE-3D convex collision meshes. Convex hulls are coarser than the artist meshes but are triangle geometry an engine can take.
- Materials: none shipped. They would come from labels. RLR's material file already works that way (below); SoundSpaces 2.0 did the same for Matterport3D (`verified`, its Sec. 3.3: "SoundSpaces coped with this issue by defining a fixed mapping from object categories to acoustic materials, e.g., floors are always mapped to the carpet material"); GWA did it for 3D-FRONT houses with a sentence-transformer label match (`verified`, GWA abstract). `room_type` could drive per-room floor choices. This is a guess about real acoustics, so results should be stated as "with assumed materials".
- Source: the GN-Matrix goal object's box center (or any labeled object), with a dry sound clip chosen by category.
- Listener: the agent pose the simulator already tracks, at the fixed 1.3 m height, converted with the simulator's own world-to-splat transform so audio and image share coordinates.
- Not usable for acoustics: the splats themselves (no surfaces) and the occupancy map (2D only).

#### 4. What adding spatial audio would take, by tool

**RLR-Audio-Propagation, standalone** (https://github.com/facebookresearch/rlr-audio-propagation, commit `4fd446b4abb5c71fb7a232a083bbddd65f25fc6f`, archived, last push 2022-11-01; `verified` from README and `RLRAudioPropagationPkg/headers/RLRAudioPropagation.h`):
- Ships "the binaries and headers", Linux x64 only (`libRLRAudioPropagation.so`), "CC-BY-NC licensed".
- The C API takes geometry from any source: `RLRA_SetObjectBox` ("Replace the mesh for an object with a rectangular box-shaped mesh", with a material for each of the 6 faces), `RLRA_SetObjectMeshOBJ` / `RLRA_SetObjectMeshPLY`, and raw `RLRA_AddMeshVertices` / `RLRA_AddMeshIndices` ("with the specified material category name"). Sources and listeners are set by position and quaternion; `RLRA_Simulate` computes "IRs for all pairs of sources and listeners"; `RLRA_GetIRChannel` and `RLRA_WriteIRWave` return them.
- Listener layouts: `RLRA_ChannelLayoutType_Mono`, `_Binaural` ("spatializes audio using an HRTF or ATF"), `_Ambisonics`; `RLRA_SetListenerHRTF` loads a custom HRTF. So mono and binaural versions of the same episode can come from the same engine and geometry, which is what a spatial-versus-mono comparison needs.
- Materials: `RLRA_SetMaterialDatabaseJSON`; a category is matched to "the material which has the greatest number of label substring matches. A match is counted if the lowercase category name contains a label as a substring." The shipped `mp3d_material_config.json` has 30 materials with labels such as `wall` (Gypsum Board), `window` (Glass), `chair`, `door` (wood, Thick), `floor` (Carpet), `ceiling` (Acoustic Tile). InteriorGS labels like "Window", "Chair", "Curtain" would match these directly after lowercasing (`speculative`, not run).
- Extra work (`speculative`): a geometry exporter (structure.json walls plus boxes or collision meshes to triangles, in splat coordinates); a thin caller for the C library (ctypes or a small C++ program; no Python binding ships with the standalone package); a per-step loop that renders the IR at each agent pose, convolves the dry clip, and crossfades between steps (SoundSpaces 2.0 Sec. 3.2 describes this: "we apply linear crossfading ... to smooth out the transition", `verified`); a WNM-3D-side audio input (the WNM-3D server takes no audio, `recalled` from the 2026-11-18 survey). Linux only, so cluster or cloud, not the Mac.

**RLR through habitat-sim's AudioSensor (SoundSpaces 2.0 route)** (`verified` from `SoundSpaces2.md` in facebookresearch/sound-spaces and `src/esp/sensor/AudioSensor.cpp` in the habitat-gs fork): the sensor takes the scene's joined semantic mesh with category-to-material mapping if a semantic scene exists, otherwise "the normal render mesh without any material info" with the default material. So inside Habitat it needs a triangle scene mesh loaded as the stage. facebookresearch/habitat-sim is archived (GitHub API, last push 2026-07-21).

**Habitat-GS** (arXiv 2604.12626 v2, ECCV 2026; https://github.com/zju3dv/habitat-gs at `83878813a9ceab815879150d4457ed9ccd0e218d`, pushed 2026-11-12; `verified` from README, `build.sh`, `docs/AUDIO.md`): a habitat-sim fork that renders Gaussian splats, keeps Habitat's NavMesh and Habitat-Lab integration, keeps the AudioSensor source and a `--with-audio` build flag (`HABITAT_WITH_AUDIO:=OFF` by default), and its scene dataset already includes InteriorGS scenes (README: "expanded with 64 InteriorGS scenes"; the folder listing says "InteriorGS (55 scenes): 3DGS and navmesh only"). The paper text has zero matches for audio or sound. Because its InteriorGS scenes carry no mesh, the audio sensor would have no geometry until a mesh is added (`speculative`, not tested). This is a second route to "Gaussian-splat visuals plus RLR audio", but it is not WNM-3D's simulator.

**pyroomacoustics** (https://github.com/LCAV/pyroomacoustics, MIT, latest release v0.10.1, master `08d16968f8db414affaf84771a650358a7961446`; `verified` from README and examples): "Fast C++ implementation of the image source model and ray tracing for general polyhedral rooms", "Convex and non-convex rooms"; `examples/room_from_stl.py` builds a room with "one wall per triangle" and notes "the materials need to be set in the program which is not very practical when different walls have different materials"; `examples/directivities/simulate_binaural_recording.py` makes binaural RIRs from SOFA HRTFs (MIT KEMAR files ship in the package). Input it needs: triangles or wall polygons with absorption and scattering per wall. structure.json could feed it directly. Pip-installable and cross-platform. Whether it models diffraction or transmission through walls was not checked; I recall it does not (`recalled`), which matters for "sound from the next room".

**Neural acoustic fields on Gaussian splats** (`verified` from arXiv HTML): these learn a scene's sound field from existing audio, so they cannot create training data for a scene that has none.
- AV-GS (arXiv 2406.08920 v3, NeurIPS 2024): a 3D Gaussian model plus an acoustic field network and binauralizer; trained and evaluated on RWAVS (recorded binaural audio) and SoundSpaces RIRs (Sec. 4.1).
- AudioGS (arXiv 2604.08967 v1): "a novel visual-free framework that explicitly encodes the sound field as a set of Audio Gaussians"; evaluated on Replay-NVAS (real recordings).
- NAMS (arXiv 2509.17410 v2): multipoles "inspired by" splatting, not Gaussian splats; trained on MeshRIR (measured) and on Treble-simulated apartments (Sec. 3.1).

#### 5. Has anyone rendered spatial audio or room impulse responses on Gaussian-splat scenes?

- Learned rendering on splats from recorded or simulated audio: AV-GS, AudioGS (above).
- Generated splat scenes with spatial audio: SonoWorld (arXiv 2603.28757 v1, abstract and section list read): from one image, lifts a panorama into "3D Gaussian splats", "places language-guided sound anchors, and renders ambisonics for point, areal, and ambient sources". Its appendix list shows reverberation through "Late reverberation parameterization from RT60" (appendix G heading), so it is not a geometric propagation simulation (`speculative` beyond the headings; the appendix body was not read).
- Gaussian-splat navigation simulators (GN-Bench, SAGE-3D, Habitat-GS): none renders audio (above).
- Audio for world models built on SoundSpaces 2.0 meshes, not splats: AudioWorldSim (arXiv 2608.21075 v1, "Built as a custom extension of Meta's SoundSpaces 2.0 platform", scenes from Matterport3D and Replica) and BinauralVAE (arXiv 2609.06837 v1, trained on AudioWorldSim data). SAVN-CE (arXiv 2603.19660 v1, CVPR 2026) moves semantic audio-visual navigation to continuous movement, also on mesh scenes. Abstracts plus keyword passes only.
- Not found: geometric acoustics (RLR, pyroomacoustics or any ray or image-source engine) run on InteriorGS, SAGE-3D or GN-Bench scenes; audio-visual navigation in any Gaussian-splat simulator.

### Entries

##### li2026gn0 / GN-Bench-Tools: the simulator code
- Links: https://github.com/TeleHuman/GN0 (master `c92778118542bf1dd7d437855d06e0d822e8361d`, Apache-2.0, pushed 2026-09-01); submodule https://github.com/TeleHuman/GN-Bench-Tools (pinned `0ce83677532591c582b0c8a2879ab431a1efff18` = master, MIT, pushed 2026-08-23); paper arXiv 2606.03682 v1 (HTML).
- Read depth: GN-Bench-Tools full file list; `README.md`, `requirements.txt`, `setup.py`, `gsplat_renderer.py` (rendering functions), `GN_Bench_simulator.py` (sensor classes, `reconfigure`, `_load_initial_state`, `step` start, function list); GN0 `README.md` (components, data layout), `gn0/benchmark/dataset.py`; keyword grep of all four repositories; paper keyword pass and Sec. 4.2.1.
- Does: Habitat-Lab-derived task and config layer over a gsplat renderer; RGB, depth, BEV; discrete VLN-CE actions or direct pose actions; 2D occupancy-grid planning and collision.
- Does not: audio of any kind; load meshes, `labels.json` or `structure.json`.

##### huang2026wnm: WNM-3D repository
- Link: https://github.com/TeleHuman/WNM-3D (master `7fe2d5f93a8dc4e2a5a2ec6fe174746928ecdcbb`, pushed 2026-09-01); paper arXiv 2608.07267 v3 (HTML).
- Read depth: keyword grep of the tarball; README data layout and acknowledgements; paper keyword pass.
- Does not: audio anywhere (only `torchaudio` install pins and `"has_audio": False` in converted video metadata). The 2026-11-18 survey already recorded that its inference server takes no audio (`recalled`).

##### TeleHuman/GN0-VLN-CE
- Link: https://github.com/TeleHuman/GN0-VLN-CE (main `ac4ebedb4ec88151656c3caa7cbcb93095497b6a`). Read depth: keyword grep only. Habitat v0.1.7 harness for GN-BAE (`recalled` from the 2026-11-18 survey); no audio hits beyond the GN-BAE processor flag.

##### spatialverse/InteriorGS
- Link: https://huggingface.co/datasets/spatialverse/InteriorGS (revision `334dfeea…`, gated auto). Read depth: API metadata and full file list; public dataset page text (overview, structure, data description, citation). Raw README and payload files refused without access ("Access to dataset spatialverse/InteriorGS is restricted"), so not read; no workaround tried.
- Does: splats, labeled oriented boxes, occupancy maps, floor plans with walls, heights, door and window openings.
- Does not: triangle meshes, materials, audio.

##### TeleEmbodied/GN-Matrix
- Link: https://huggingface.co/datasets/TeleEmbodied/GN-Matrix (revision `50a68253…`, not gated, CC BY-NC 4.0). Read depth: API file list, empty card, datasets-server schema and first row, GN0 loader.
- Does: start pose, goal position, goal object id and label, instructions, reference path. Does not: any sound field or source annotation.

##### miao2026towards: SAGE-3D and its collision meshes
- Links: arXiv 2510.21307 v2 (HTML; abstract, Sec. 1, 2.2, 2.3 read), https://github.com/Galery23/SAGE-3D_Official README (scene data preparation), https://huggingface.co/datasets/spatialverse/SAGE-3D_Collision_Mesh (card, API file list, one file size), https://huggingface.co/datasets/spatialverse/SAGE-3D_InteriorGS_usdz (API file list).
- Does: per-object CoACD convex collision bodies for all 1,000 InteriorGS scenes, USD format, Isaac Sim 5.0+ environment with RGB, depth, semantics, contacts.
- Does not: audio; acoustic materials. Contents of the USD files not inspected (binary).

##### facebookresearch/rlr-audio-propagation
- Link above, commit `4fd446b4…`. Read depth: README, file tree, full header function list and the object, material and simulation sections, `mp3d_material_config.json` material names and labels.
- Does: geometry-agnostic acoustic ray tracing from boxes or triangle meshes, label-matched materials, mono/binaural/ambisonic IRs. Does not: run on anything but Linux x64; ship a standalone Python binding.

##### chen2022soundspaces2: SoundSpaces 2.0 paper and API doc
- Links: arXiv 2206.08312 v2 (HTML; Sec. 3.3 material modeling, 3.4 generalization, 8.4 material configuration); `SoundSpaces2.md` in https://github.com/facebookresearch/sound-spaces (full).
- Does: "Our new simulator accommodates arbitrary 3D meshes as input"; label-to-material mapping; shoebox APIs; crossfading for continuous motion. Relevant here as the engine, not the scenes.

##### xia2026habitatgs: Habitat-GS
- Links: arXiv 2604.12626 v2 (HTML, keyword pass), https://github.com/zju3dv/habitat-gs (README, `build.sh`, `docs/AUDIO.md`, `src/esp/sensor/AudioSensor.cpp` mesh-loading lines; GitHub code search for "audio": 55 hits; the first 15 paths shown are the inherited audio-sensor source, its docs and tutorial, and build files; the rest were not listed).
- Does: splat rendering inside habitat-sim; InteriorGS scenes with NavMesh; audio sensor code present but off by default. Does not: ship meshes for its InteriorGS scenes; mention audio in the paper.

##### LCAV/pyroomacoustics
- Link above. Read depth: README (features), `examples/room_from_stl.py`, start of `examples/directivities/simulate_binaural_recording.py`, repository tree for STL, SOFA and directivity files.

##### bhosale2024avgs (AV-GS), AudioGS 2604.08967, NAMS 2509.17410, SonoWorld 2603.28757, GWA 2204.01787, AudioWorldSim 2608.21075, BinauralVAE 2609.06837, SAVN-CE 2603.19660
- arXiv HTML, abstracts plus the sections named above; keyword passes. Versions as listed in Sec. 4 and 5. None renders audio in GN-Bench or InteriorGS.

### Searches run

- GitHub API tarballs at the commits above, grep (case-insensitive) for `audio|sound|acoustic|\brir\b|\.wav|binaural|ambison|reverb|impulse`: GN-Bench-Tools 0 hits; GN0, WNM-3D, GN0-VLN-CE hits listed in Answer 1.
- Grep for `\.ply|\.json|structure|labels|occupancy|mesh` in GN-Bench-Tools Python and for `interiorgs|gn-matrix|labels.json|structure.json|occupancy` in GN0 and WNM-3D: the simulator opens only `occupancy.json`, `occupancy.png`, `3dgs_compressed.ply`.
- arXiv HTML keyword passes (audio, sound, acoustic, binaural): GN0 2606.03682 v1, WNM-3D 2608.07267 v3: 0 hits each; Habitat-GS 2604.12626 v2 (audio, sound): 0 hits.
- Hugging Face API: spatialverse/InteriorGS, TeleEmbodied/GN-Matrix, spatialverse/SAGE-3D_Collision_Mesh, spatialverse/SAGE-3D_InteriorGS_usdz (metadata, file lists); datasets-server first-rows for GN-Matrix; one HTTP range request (first 256 KiB) of `Collision_Mesh/839920/839920_collision.usd`. InteriorGS raw README refused (gated); not worked around.
- GitHub: rlr-audio-propagation (README, tree, header, material JSON), sound-spaces `SoundSpaces2.md`, habitat-sim metadata, habitat-gs (README, build.sh, docs/AUDIO.md, AudioSensor.cpp, code search "audio" and "navmesh"), pyroomacoustics (README, tree, two examples), SAGE-3D_Official README.
- Web searches (used only to find sources, each hit opened before use): "Towards Physically Executable 3D Gaussian for Embodied Navigation arXiv"; "AV-GS audio-visual Gaussian splatting novel view acoustic synthesis"; "Gaussian splatting room impulse response acoustic rendering"; "rlr-audio-propagation github mesh material"; "SoundSpaces 2.0 arbitrary mesh custom scene audio sensor habitat-sim"; "binaural audio Gaussian splatting scene simulation embodied navigation 2026"; "InteriorGS audio sound simulation" (nothing relevant); "SAGE-3D InteriorGS acoustic room impulse response" (nothing relevant); "\"Gaussian splatting\" simulator \"room impulse response\" navigation audio-visual embodied"; "GN-Bench GN0 audio navigation Gaussian splatting" (no audio work); "GWA geometric wave acoustic dataset 3D-FRONT room impulse responses arXiv"; "audio-visual navigation 3D Gaussian splatting scenes binaural rendering mesh extraction".
- Leads seen but not opened (so not used): AV-Cloud (NeurIPS 2024), "Hearing Anywhere in Any Environment" (arXiv 2504.10746), NavGSim (arXiv 2603.15186), a GN-Bench-Tools fork under another GitHub account, AV-Twin (arXiv 2512.10778; abstract opened only: smartphone-measured RIRs plus material estimation, real rooms, not splat simulators).
- Not done: running any engine; opening the gated InteriorGS files; reading the collision USD contents; the CVPR 2026 SAVN-CE PDF (the arXiv version was used).

## Part B: the two papers the PI linked

Date: 2026-11-19. Read by the literature session. Nothing in this file is committed; it is a working note for the lead.

### Answer block
1. **Hear, Localize, and Reason (HLR-LLM / HLR-AVSceneQA), arXiv 2610.21588v1.** Read in full text (arXiv HTML: main text, sections 3 to 6, appendix A.1, A.3, C.1, C.5, F; appendix B and C.2 to C.4 only skimmed by search). A question-answering model, not an acting model. Spatial audio is made with SoundSpaces 2.0 inside Habitat-Sim on HM3D scenes (Habitat only, no Gaussian splats), rendered as 2-channel binaural at 32 kHz, one static viewpoint per sample, one image plus audio per question. The spatial branch is a frozen Spatial-AST encoder (from BAT) plus a trainable MLP projector, added to Qwen2.5-Omni next to its monaural audio path. Dataset: 175,061 train and 21,808 test QA pairs. The project page exists but its Code and data buttons were placeholders ("#") when read, and no code or data repository was found.
2. **Hear You Are (CVPR 2026), Ryu, Chung, Harwath.** Read in full: the open-access PDF (this session's plain download returned HTTP 200, unlike the earlier fetch) and most of the supplement. Not BLOCKED. Same family of idea: SoundSpaces 2.0 on Matterport3D, binaural at 32 kHz, 10-second VGGSound clips, 360 degree panorama image, 1 million QA pairs, frozen Spatial-AST audio encoder, Q-Former projectors, Qwen2-7B-Instruct with LoRA. Code, dataset and checkpoints are promised "upon acceptance"; no public repository or Hugging Face entry was found.
3. **Neither renders audio in Gaussian-splat scenes or any non-Habitat scene.** Both use SoundSpaces 2.0 on Matterport-family meshes (HLR says Habitat-Sim explicitly; Hear You Are does not name Habitat, so that part is recalled). Neither handles long audio or a moving agent: both are one short clip and one static pose per question, and HLR lists dynamic agents and moving sources as future work.
4. **What it means for us (speculative, the literature session's reading).**
   - Method to reuse: the spatial audio branch recipe (frozen Spatial-AST plus small projector, aligned first on audio-only spatial QA, then trained with the language model) is the shared pattern of both papers and the closest published template for "add a spatial audio encoder to a big pretrained model". It was built for a language model, not for a video diffusion or world-action model like Wan 2.2.
   - Data to reuse: not directly. The data is SoundSpaces on Matterport/HM3D, which is what the PI said not to use if WNM-3D's own simulator can do audio. Neither dataset is public yet. HM3D and Matterport terms restrict redistribution.
   - Neighbors to cite: yes, both, for question 1. Both contain a mono versus binaural comparison with vision kept in (Hear You Are Table 4; HLR Table 5 row c). Hear You Are shows the gain from binaural audio is large when the source is not visible and small when it is, which is the target-visibility split already on the candidate list. Neither tests this inside a world model; neither predicts frames or actions.
   - Possible angle for the PI's long-form question: both papers use one short clip and one static pose. That is a statement about these two papers only, not a survey of audio-VLA work (that is Task C).

### Source 1: arXiv 2610.21588, "Hear, Localize, and Reason: Spatially Aware Scene Understanding for Audio-Visual LLMs"
- Link: https://arxiv.org/abs/2610.21588 , HTML https://arxiv.org/html/2610.21588
- Version: v1 (`verified`: "arXiv:2610.21588v1 [cs.CV] 15 Oct 2026"). Venue: NeurIPS 2026 per the arXiv comments field (as recorded in the 2026-11-19 LOG entry) and the project page. Authors: Kim Sung-Bin, Lee Jung-Mok, Jinwoo Jung, Oh Hyun-Bin, Hyeonggon Ryu, David Harwath, Tae-Hyun Oh. Tier 2 (arXiv preprint, not peer reviewed as far as the arXiv listing shows).
- Read depth: read (main text in full; appendix partly). Untrusted copy: scratchpad downloads/B/hlr.html (not executed).

#### What it does
- `verified` (Abstract): "a comprehensive benchmark with a hear-localize-reason hierarchy" and "HLR-LLM, which augments a strong audio-visual foundation model with a binaural spatial audio branch alongside its monaural semantic audio pathway".
- Task: question answering over one rendered image plus binaural audio: 12 tasks in three families (spatial hearing: sound event, far-field speech; spatial grounding: egocentric and allocentric direction and distance; spatial relational reasoning: visibility, correspondence, hidden-source, loudness change, proximity, scene captioning) (Section 3, Table 2).
- Inputs: one RGB frame ("a 90 degree horizontal field of view at 1280 x 720"), mono audio (channel average of the binaural signal), binaural audio, and a text question. Outputs: text (mostly multiple choice or binned values; speech transcription and captioning are open-ended) (Sections 3.1, 3.2, 4.3).
- Not a navigation or acting model, and no video generation.

#### How the spatial audio data was made (`verified`, Sections 3.1, 4.3, appendix C.1)
- Simulator: SoundSpaces 2.0 "renders geometry-aware room acoustics and spatializes dry monaural clips into binaural audio"; appendix C.1: "We generate each sample in Habitat-Sim with acoustic rendering using the SoundSpaces simulator".
- Scene source: HM3DSem, "216 indoor spaces and 3,100 rooms"; "145 scenes for training and 36 for testing".
- Sound sources: VGGSound non-speech clips (classes manually filtered, ImageBind text-audio similarity filter) and LibriSpeech for speech. Objects: Objaverse-XL meshes plus generated meshes (Stable Diffusion 2, then TRELLIS), "approximately 81 meshes per category for training and 35 for testing". 2 to 6 objects inserted per room, up to two active sound sources.
- Rendering method: convolve the loudness-normalised dry clip with the SoundSpaces binaural room impulse response, sum over sources (appendix C.1, equations 1 to 3).
- Audio format: binaural, 2 channels, 32 kHz (Section 4.3). Not Ambisonics, not a mic array.
- Clip length in seconds: **not found** in the parts read. The spatial encoder follows BAT preprocessing ("window size of 1024, a hop size of 320, and 128 mel bins"). The only stated 10-second clips are the real STARSS23 clips (appendix A.3: "non-overlapping 10-second clips"). VGGSound clips are 10 seconds per the Hear You Are paper, but HLR does not say whether it trims them. Treat the length as unknown.
- Viewpoint: one agent pose per sample, placed "near a room corner and orient[ed] toward the room center", with some poses nearer the centre (appendix C.1). No trajectories.
- Dataset size (Section 3): "175,061 training and 21,808 test QA pairs, including 53,279 training and 4,869 test pairs from scenes with multiple simultaneously active sound sources. We also provide 94,519 CoT-augmented training pairs". Chain-of-thought text is generated by templates from simulator metadata; GPT-4 only paraphrases question wording.
- Gaussian-splat or non-Habitat scenes: **no**. HM3D in Habitat-Sim only (`verified`, C.1). The one real-data test is STARSS23 recordings (Section 5.4), where the audio is real, not rendered.

#### How the spatial audio branch is built and trained (`verified`, Sections 4.1 to 4.3, appendix A.1)
- Base model Qwen2.5-Omni. New spatial branch: pretrained Spatial-AST encoder (from BAT), then a "lightweight" linear-GELU-linear MLP projector, with a new media token `<spatial_audio>`. Input sequence X = [text, visual tokens, mono audio tokens, spatial audio tokens].
- Pretraining: stage 1 aligns the projector (and resampler) on BAT's spatial audio-only QA data (514k samples); stage 2 instruction-tunes on HLR-AVSceneQA (175k); stage 3 continues with chain-of-thought answers (94.5k).
- Frozen or not: "All visual, monaural audio, and spatial audio encoders remain frozen throughout training." Stage 1 trains only the projector; stages 2 and 3 train the projector and LoRA on the language model (rank 16).
- Compute: 2 epochs on four A6000 GPUs. Results are mean and standard deviation over three runs.
- Mono versus binaural evidence (Table 5, three runs): removing binaural audio (row c, vision plus mono) drops the grounding numbers a lot. The numbers in row (c) are identical to the fine-tuned Qwen2.5-Omni row in Table 3 (32.69, 30.92, 26.26, 27.73), so it is the same model. A dual-channel Qwen2.5-Omni (left and right sent through the ordinary audio path, no spatial branch) already gets much of the gain, and the explicit spatial branch beats it on every grounding metric (Table 3). So part of the benefit comes from simply feeding two channels.

#### What is released
- Project page https://hlrllm.github.io/ : the abstract says "Code and benchmark data are available at https://hlrllm.github.io/". The page source has four buttons (arXiv, Paper, Code, HLR-AVSceneQA) whose href is "#" on 2026-11-19, so no links yet. Page footer: Creative Commons Attribution-ShareAlike 4.0 (applies to the web page).
- GitHub: the only repository under the HLRLLM organisation is hlrllm.github.io (latest commit 8c0031fff1c95399409a0ca9438b518f97592d80, 2026-10-15). No model code, no weights.
- Hugging Face: searches for HLR-LLM, HLR-AVSceneQA, HLRLLM returned nothing (API, 2026-11-19).
- Licenses (appendix F, `verified`): SoundSpaces released under CC BY 4.0; Matterport3D-based task data and trained models "follow the Matterport3D terms and CC BY-NC-SA 3.0 US license"; HM3D under academic research terms; "We will release only assets and annotations that are compatible with the applicable licenses and redistribution terms." So the dataset, if released, will likely be non-commercial.
- Upstream pieces (gh api, 2026-11-19): zszheng147/Spatial-AST (latest commit a5638035e9a4dd3da3d4a23873af719062710586) has a LICENSE file whose text is "Attribution-NonCommercial 4.0 International" (GitHub's license field says NOASSERTION); QwenLM/Qwen2.5-Omni is Apache-2.0; facebookresearch/sound-spaces license field is CC-BY-4.0.

#### Limitations stated by the authors (Section 6)
"HLR-AVSceneQA is built with a simulator-based pipeline ... The benchmark focuses on indoor scenes, leaving dynamic agents, moving sound sources, and richer embodied interaction for future work."

### Source 2: CVPR 2026, "Hear you are: Teaching LLMs Spatial Reasoning with Vision and Spatial Sound"
- Links: https://openaccess.thecvf.com/content/CVPR2026/html/Ryu_Hear_you_are_Teaching_LLMs_Spatial_Reasoning_with_Vision_and_CVPR_2026_paper.html ; PDF https://openaccess.thecvf.com/content/CVPR2026/papers/Ryu_Hear_you_are_Teaching_LLMs_Spatial_Reasoning_with_Vision_and_CVPR_2026_paper.pdf ; supplement https://openaccess.thecvf.com/content/CVPR2026/supplemental/Ryu_Hear_you_are_CVPR_2026_supplemental.pdf
- Version: CVF open-access version, "identical to the accepted version" except the watermark; CVPR 2026 proceedings pp. 38606 to 38615. Authors: Hyeonggon Ryu (Hankuk University of Foreign Studies), Joon Son Chung (KAIST), David Harwath (UT Austin). Tier 1 (peer reviewed).
- Access note: the brief said a plain fetch got HTTP 403 earlier. This session a plain download of the landing page, PDF and supplement each returned HTTP 200, with no workaround. Untrusted copies: scratchpad downloads/B/ryu.pdf and ryu_supp.pdf (text extracted with pdftotext, not executed).
- Read depth: read (main paper in full; supplement 8.1 and 8.2 plus figure captions). An arXiv version was searched (API rate-limited or 503; arXiv web search showed no hit); none found.

#### What it does
- `verified` (Abstract, Section 1): new task "Audio-Visual Spatial Reasoning", the dataset Hear You Are QA, and the model "Hear You Are LLM" connecting a spatial audio encoder and a vision encoder to a language model.
- Task: question answering over one 360 degree panorama plus 10 s binaural audio. Nine question types (Table 1): spatial correspondence, relative location (distance, direction, angle), spatial and semantic correspondence (single or multiple matching objects), semantic co-occurrence (is the source visible). Short text answers.
- Inputs: panorama (224 x 812), binaural waveform, question text. Output: text. Not a navigation or acting model.

#### How the spatial audio data was made (`verified`, Sections 3 and 5.1)
- Simulator: SoundSpaces 2.0 "renders geometry-based acoustics, adding realistic reverberation for any source-receiver pair". Habitat is not named in the text read; that SoundSpaces 2.0 runs on Habitat is `recalled`.
- Scene source: Matterport3D, 90 scenes, "72 scenes for training, 9 for validation and 9 for testing".
- Rendering: monaural clip convolved with the room impulse response; receiver "record[ing] a binaural audio signal with the default Head Related Transfer Function (HRTF) provided by SoundSpaces2.0".
- Audio format: binaural. Clip length: "a 10-second audio binaural waveform sampled at 32 kHz" (Section 5.1). Source clips: VGGSound, "200,000 in-the-wild 10-second YouTube clips", outdoor and ambiguous classes removed.
- Objects: images from Stable Diffusion 3, lifted to 3D with InstantMesh; 150 VGGSound classes and 40 ImageNet classes; up to three random distractor objects. One sound source per scene. Image: 18 views of 20 degrees stitched into a 360 degree panorama.
- Dataset size: "1 million question-answer pairs" (Section 1). Questions paraphrased with ChatGPT-4o.
- Gaussian-splat or non-Habitat scenes: **no**, Matterport3D meshes with SoundSpaces only.

#### How the spatial audio branch is built and trained (`verified`, Sections 4 and 5.1)
- Audio encoder: pretrained Spatial-AST binaural encoder from BAT, "kept frozen throughout the entire training process".
- Projectors: Q-Former for both modalities, 64 query tokens for audio and 128 for vision; the audio projector uses BAT's implementation and pretrained weights.
- Vision: SigLIP2 NaFlex with LoRA. LLM: Qwen2-7B-Instruct, LoRA rank 16.
- Pretraining ("Warm Start"): each encoder and projector is first trained alone on classification and localisation questions built from simulator metadata; the visual side in two steps (classification, then localisation).
- Compute: "3 epochs on 8 A5000 GPUs with an effective batch size of 128 ... The training takes three days." No seed count stated.
- Mono versus binaural evidence (Table 4, one run each as far as the paper states): vision plus binaural beats vision plus mono mainly when the source is not visible. Q4 with invisible audio, direction accuracy: R+B+Q 41.18 against R+M+Q 16.71. With visible sources they are close (Q4 direction accuracy 65.68 against 59.03; Q1 sound class 52.69 against 51.01).

#### What is released
- Paper statements: "We will open source both the dataset and the training code" (Section 1); supplement 8.1.3: "We will release the full codebase, panoramic image dataset, reverb files, model checkpoints, and detailed instructions for reproducing all experiments upon acceptance." The paper is now in the CVPR 2026 proceedings, so release is due.
- Searched on 2026-11-19, nothing found: GitHub repository searches (see below), a GitHub user lookup for the first author, Hugging Face model and dataset searches for "hear-you-are", the first author's Google Sites page (affiliation only, no project link). The CVF landing page lists only pdf, supp and bibtex. No license is stated in the paper; the underlying assets have their own terms (Matterport3D, VGGSound, Stable Diffusion 3).

#### Limitations
One sound source per scene, static agent, 10 s clips, simulator only, no real-recording evaluation, no seed count reported.

### How the two relate (`verified`, HLR Sections 1 and 5.1)
HLR cites Hear You Are as Ryu et al. (2026) and uses it as its closest baseline; the two share two authors (Ryu, Harwath). HLR argues for normal field of view over panorama input and adds multi-source scenes, speech, hidden sources and chain-of-thought training. HLR's fine-tuned Hear You Are baseline is weaker than HLR-LLM on all grounding tasks (Table 3), but the HLR authors re-trained it themselves on their own data, so it is an in-house comparison.

### What it means for this direction (speculative, planning input only)
- **For the PI's WNM-3D question (Task A):** these papers do not answer whether WNM-3D's simulator can produce audio. They give a checklist of what a usable set needs: a room impulse response per source-receiver pair, binaural or multichannel output, dry source clips with sound classes, sources both visible and hidden. Whether WNM-3D can meet it is Task A.
- **For question 1 (spatial against mono against none in a world action model):** both papers support the expected direction (binaural helps, much more when the source is not visible) in language-model question answering. Neither tests a world model, so question 1 stays open. HLR's dual-channel control (left and right through the existing mono path, compared with a dedicated spatial encoder) is worth copying.
- **For the PI's long-form suggestion:** both papers sit at the short end (10 s or less, one static viewpoint). Whether the wider audio-VLA literature leaves this open is Task C.
- **Not verified:** that either project's code or data will appear; the clip length used in HLR-AVSceneQA; HLR appendix B and C.2 to C.4.

### Searches run (2026-11-19)
- arXiv HTML https://arxiv.org/html/2610.21588 (200, full text read).
- CVF landing page, PDF and supplement for the Ryu paper (all 200).
- arXiv API title search for "Hear you are" (503, then rate exceeded) and arXiv web search for the title: no arXiv version found.
- GitHub (gh search repos): "Hear You Are", hear-you-are-llm, "Hear Localize Reason", HLR-LLM, hlrllm, "audio-visual spatial reasoning Hyeonggon Ryu", "spatial audio visual question answering SoundSpaces Qwen2-7B Spatial-AST": only unrelated repositories plus HLRLLM/hlrllm.github.io. gh api orgs/HLRLLM/repos: one repository. gh search code "HLR-AVSceneQA": only third-party digests of the abstract (not used as sources).
- Hugging Face API model and dataset searches: HLR-LLM, HLR-AVSceneQA, HLRLLM, hear-you-are: empty.
- First author's Google Sites page: no project link.
- Upstream licenses via gh api: Spatial-AST, sound-spaces, Qwen2.5-Omni.
- Keyword search (Gaussian, splat, 3DGS, NeRF, navigat, long-form) in both full texts: no relevant hit; HLR names "dynamic agents, moving sound sources" as future work only.

## Part C: what existing audio-VLA studies cover and leave out

Date: 2026-11-19. Direction: hear-lab. Written by a worker agent for the literature session (read-only on the repo; nothing committed). Question from the PI (via the planning lead's 2026-11-19 PI-feedback LOG entry): "it would be a good idea to pursue research in a direction that addresses aspects not covered in existing audio-VLA studies (e.g., long-form processing)".

How this was done: started from `hear-lab/literature/survey/2026-11-16-audio-in-robot-policies.md` and the NaVLA2 entries in the other survey files (`git show origin/main:...`, plus the 2026-11-18 novelty file on the current branch), then searched for audio-VLA papers missing there. Every paper below was downloaded in full (arXiv HTML, a PDF read with `pdftotext`, a GitHub tarball, or a project page) into the scratchpad and read through targeted extraction of the method, setup, ablation and limitation passages (not every paragraph); numbers and quotes are copied from those texts. Page or section names are given. No AI-generated summaries or search snippets were used as evidence. Labels: `verified` (read in the named place), `recalled` (from the survey files or memory, not reopened this session), `speculative` (my reading).

### Answer block

1. **Audio window length.** Among the audio-VLA and audio-conditioned policies I could open, the audio the policy hears at one decision step is short: about half a second (ADAPT), 1 s (S2A2), 2 s (ListenVLA, ManiWAV), 4 s (HEAR), 7 s mono plus 10 s two-channel (NaVLA2, from its code), and the longest, RoboOmni, trains on dialogue clips of "10 to 80 seconds" that are given as an instruction, not listened to over an episode. `verified`.
2. **Instruction or stream.** Two groups. One-off instruction: VLAS, Speech-pi0.5, RoboOmni (speech and sound dialogue before acting). Ongoing sound: HEAR (streaming, with a causal memory), ListenVLA (a 2 s rolling history), S2A2, ManiWAV, SoundAct, Audio-VLA, ADAPT, NaVLA2 (new audio each step). `verified`.
3. **Memory over time.** Only HEAR has an audio memory that carries across decisions (a "Streaming Stateful Transformer"). ListenVLA keeps a 2 s history. ADAPT keeps the last eight actions, not audio. NaVLA2 keeps past video frames, not past audio. Navigation systems with a persistent memory (LH-AVLN's PAG-Nav) store acoustic cues in a map and are not VLAs. `verified`.
4. **Spatial versus one channel.** Spatial: NaVLA2 (binaural plus a mono-downmix branch), ADAPT (first-order Ambisonics), S2A2 (microphone arrays), SoundAct (stereo), MuseVLA (6+1 microphone array turned into a sound heatmap). One channel: HEAR, ListenVLA (simulated "virtual monaural microphone"), ManiWAV, Audio-VLA (contact microphones), VLAS and RoboOmni (speech). `verified`, with the exceptions in the table.
5. **Navigation or manipulation.** Navigation: NaVLA2, ADAPT (simulation only), BCNav (not a VLA), LH-AVLN (benchmark plus a training-free agent). Everything else I found is manipulation (EchoVLA is driving, open-loop). `verified`.
6. **Real or simulated.** All navigation audio-VLA work I found is simulation only (NaVLA2, ADAPT, LH-AVLN); the one real-robot navigation result is BCNav (a DOA estimator plus depth policy, not a VLA). Real-robot audio-VLA results exist for manipulation (HEAR, ListenVLA, Audio-VLA, ManiWAV, SoundAct, S2A2 with ACT only, MuseVLA, FuSe). `verified`.
7. **Long-form audio.** In the audio literature "long-form" means minutes to hours (ASR: utterances "that last from minutes to hours"; understanding benchmarks: 90 s to 300 s at the short end, 10 minutes to over 2 hours, an average of 51 minutes, up to six days). Short clips are "typically under 30 seconds". Audio language models degrade as audio grows. Details and sources in the long-form section. `verified`.
8. **What none of the audio-VLAs does** (search result, not a proof): none processes minutes-to-hours audio; none evaluates a policy on audio length as a variable except ManiWAV's 1 s and 10 s ablation (10 s was worse); none has an audio memory whose span is measured against episode length; none combines spatial audio with a world model that predicts video and actions. Full list below. Speculative: of these, the fits for a spatial-audio world action model are listed at the end.

Not opened (do not rely on): Hear-to-Act, HearVGo, D2AVN, TraSNav (OpenReview PDFs return HTTP 403; I did not try to get around it), and the CVPR 2026 "Hear You Are" paper (HTTP 403 for the planning lead; not part of this task). SoundAct's paper PDF is on OpenReview (HTTP 403 again today), so SoundAct rests on its project page.

### Table (compact)

Window = the audio the model takes at one decision step, as stated. "Stream?" = whether audio keeps arriving during the episode. All rows `verified` from the named source unless marked.

| Model | Audio window (quote or code) | Instruction or stream | Memory over time | Spatial? | Nav or manip | Real or sim |
|---|---|---|---|---|---|---|
| HEAR (2603.16086) | "a 4.0 s query-time window" | stream, listens through delays | yes: Streaming Stateful Transformer memory across packets | one channel ("a single microphone") | manipulation | sim (RoboTwin2) + real Franka |
| ListenVLA (2610.16313) | "past two seconds ... three overlapping one-second windows" | stream (causal history) | 2 s history only | one channel ("virtual monaural microphone" in sim; BOYA mini mic real) | manipulation | sim (LIBERO-Audio) + real |
| NaVLA2 (AAAI-26) | not in the paper; code: semantic branch 7 s mono at 44.1 kHz, spatial branch 10 s two-channel at 32 kHz | new audio each step; instruction can also include an "auditory description" | visual only (K past frames); no past audio | yes: binaural plus mono-downmix branch | navigation | sim only (SoundSpaces 2.0) |
| ADAPT (2610.15956) | "0.512 s observation window" | stream (new window each step) | last 8 actions, no audio memory | yes: first-order Ambisonics | navigation | sim only |
| S2A2 (2607.26047) | "a 1.0-s audio window", updated every 0.1 s | stream | observation history of 2 steps | yes: 4 arrays, 8 channels each | manipulation | sim + real (ACT only) |
| SoundAct (project page) | not stated on the page | stream | not stated | yes: stereo | manipulation | real |
| ManiWAV (2406.19464) | "the last 2-3 seconds, depending on the task"; ablation 1 s and 10 s | stream | none beyond the window | one channel (contact mic) | manipulation | real |
| Audio-VLA (2511.09958) | "fixed-length chunks corresponding to each timestep" (length not stated) | stream | none stated | one stream (two contact mics, not used spatially) | manipulation | sim (library-synthesized audio) + real |
| VLAS (2502.13508) | speech mel "padded to a fixed length of 3000 frames" | one-off instruction | none | one channel | manipulation | sim (CALVIN); real only qualitative |
| Speech-pi0.5 (2610.33866) | "N=1500 frames" of speech hidden states | one-off instruction | none | one channel | manipulation | sim (LIBERO) |
| RoboOmni (2510.23763) | dataset clips: "The majority of clips range from 10 to 80 seconds"; per-step segment length not stated | instruction-like dialogue and sounds before acting | not stated | not stated | manipulation | sim + qualitative real |
| Zhang and Gienger (2512.08405) | window length L and overlap L' defined; values not found | stream, plus predicted future audio | overlap carry-over only | not stated (one omnidirectional mic) | manipulation | real + MIDI sim |
| FuSe (2501.04693) | spectrogram, length not found | stream | not found | one channel (not stated otherwise) | manipulation | real |
| MuseVLA (2606.17598) | not found; beamformed into a 2D sound-intensity heatmap | stream | not found | yes: 6+1 mic array | manipulation | real |
| OmniEcho (2609.23407) | "the FOA spatial audio At within the current temporal window"; length not found | stream | past video keyframes | yes: first-order Ambisonics | navigation (open-loop trajectory) + QA | benchmark from real recordings plus rendered |
| BCNav (2609.37084), not a VLA | chunk length not found; array at 16 kHz | stream (bearing updated per chunk) | not stated | yes: 8-channel array reduced to one bearing | navigation | sim + real rover |
| LH-AVLN / PAG-Nav (2607.03920), not a VLA | per-step binaural audio, length not found | stream, alternating sources over a mission | yes: persistent acoustic memory in a semantic map | yes: binaural | navigation | sim only |
| Audio-Visual World Models (2512.00883), world model | "a 0.15-second binaural audio segment" per frame | stream | "context length of 4 frames" | yes: binaural | navigation | sim |
| EchoVLA, "Listen, Look, Drive" (2601.12142) | speech commands, length not found; "sliding-window attention" | stream of audio instructions | not found | one channel (speech) | driving, open-loop benchmark | nuScenes with synthetic speech |
| HLR-LLM (2610.21588), question answering | not found | one clip per question | none | yes: binaural plus mono | none (not an acting model) | sim (SoundSpaces 2.0) |

Earlier contact-microphone papers (Li 2022: 0.5 s mel clips; Du 2022: 2 s buffer; Mejia 2024: 2 s clips) are `recalled` from the 2026-11-16 survey and were not reopened.

### Per-source entries (what each does and does not do)

Read depth for all entries: full file downloaded; method, setup, ablation and limitation passages extracted and read; other sections not read. Versions are the HTML served on 2026-11-19; where the header showed a version it is given.

#### fan2026navla: NaVLA2 (AAAI-26, doi:10.1609/aaai.v40i22.38886)
- Links: PDF https://tanmingkui.github.io/files/publications/NaVLA2.pdf (read with `pdftotext`); code https://github.com/felixfjg/NaVLA at commit 436ad6b05e5fb9560f9c4e20770784924419424f (tarball grepped, never run).
- Does: binaural audio in navigation, simulated in SoundSpaces 2.0. "Both audio and visual observations are updated at each time step". History is video only: "O_{0:t-1} is the historical visual observation ... Historical visual observations are uniformly sampled K frames". The paper gives no audio duration (I searched the PDF text for duration, window, clip length and seconds).
- Code (`model/NaVLA/llava/utils.py`, lines 189-190): `audio_semantic = preprocess_audio(audio, sr, 1, 44100, 7)` and `audio_spatial = preprocess_audio(audio, sr, 2, 32000, 10)`, meaning mono, 44.1 kHz, 7 s for the semantic branch and two-channel, 32 kHz, 10 s for the spatial branch (clips padded or cropped to those lengths). `verified` for the code; whether the released code matches the trained model is not checked.
- Does not: any audio memory, any real robot, any audio longer than about 10 s.

#### nie2026towards: HEAR (arXiv 2603.16086; HTML version as served, v2 per the survey)
- Link: https://arxiv.org/abs/2603.16086
- Does: streaming audio for manipulation. Sec. 3.2.4: "a 50–200 ms event can be embedded in a 4.0 s query-time window after the robot has continued executing". Sec. 4.2 (Historizer): "a Streaming Stateful Transformer that carries a compact memory across packets", with overlapping packets and a ring buffer. Sec. 4.2 also states: "relying on a very long window alone can mix distinct interaction phases and make performance more sensitive to the exact window length." Sec. 6.4 ablation: removing the Historizer drops simulation success to 0.57; lighter memories score "a causal GRU (0.67)" and "EMA pooling ... (0.62)". Task "Boil Water" has a randomized time to boil and a "prolonged, visually static period" (Sec. 6.2).
- Does not: spatial audio ("a single microphone"), navigation, audio memory length against minutes (no span in seconds or minutes is stated for the memory in the text I read), real audio beyond scripted tasks (the survey notes the controllers are rule-based).

#### li2026listenvla: ListenVLA (arXiv 2610.16313 v1, 2026-10-12)
- Link: https://arxiv.org/abs/2610.16313
- Does: Sec. III-B1: "The past two seconds are divided into three overlapping one-second windows", each encoded by frozen CLAP plus energy cues, then a temporal encoder. Sim audio: "a virtual monaural microphone at the end effector" (benchmark section). Real tasks include waiting for sound: "Sound onset was randomized between 3 and 12 seconds after each trial began." Baselines: HEAR and ManiWAV adapted to LIBERO-Audio.
- Does not: spatial audio; any history longer than 2 s; navigation.

#### kim2026soundact: SoundAct (project page https://vision3d-lab.github.io/soundact/, fetched 2026-11-19)
- Does: stereo audio for beyond-sight manipulation; its table has stereo 10 / 10, mono (left only) 8 / 10, vision-only 5 / 10 (small trial counts). The page does not state a window length or memory. OpenReview PDF: HTTP 403 (not retried beyond one request).

#### hiratsuka2026s2a2: S2A2 (arXiv 2607.26047)
- Does: Appendix: "At each observation step, the policy receives the most recent 16,000 samples, corresponding to a 1.0-s audio window"; "The audio analysis window is 1.0 s (16,000 samples) and is updated every 0.1 s, so adjacent windows overlap by 0.9 s"; "the standard observation history of two steps". Arrays: four, 8 channels each. Sim (Genesis) plus a real ALOHA-type robot with ACT only.
- Does not: any audio memory beyond the 1 s window and two-step history; navigation; a VLA backbone in the real-robot result (pi0 appears only in simulation per the survey).

#### huang2026adapt: ADAPT (arXiv 2610.15956 v1, 2026-10-11)
- Link: https://arxiv.org/abs/2610.15956
- Does: appendix: "We use 24 frames per observation, resulting in Ta=24×1024=24576 samples, corresponding to a 0.512 s observation window"; four-channel first-order Ambisonics; Sec. 4.4: "a short history of the eight most recent navigation actions" serialized into the prompt (no audio history); the target is "a stationary mono acoustic emitter" playing a "fixed linear chirp sweeping from 500 Hz to 900 Hz over 50 ms". Limitations: "Evaluating ADAPT on a physical audio-depth robot is therefore an important direction for future work."
- Does not: a real robot, speech or semantic sounds, audio memory.

#### chen2026lhavln: LH-AVLN and PAG-Nav (arXiv 2607.03920)
- Does: "persistent spatialized acoustic cues" over a mission of "two to four goals"; "At each time step, binaural audio is rendered according to the relative pose between the agent and the currently active source." PAG-Nav "maintains persistent geometric, semantic, and acoustic memory" (a training-free map-based agent). Sim only (SoundSpaces 2.0, Matterport3D).
- Does not: train a VLA or a world model; process long audio streams (the memory stores localized cues in a map, not audio).

#### Other entries (shorter)
- **BCNav** (arXiv 2609.37084 v1): DOA from an 8-channel array reduced to one scalar bearing; "SRP-PHAT front-end dominates at 115 ms per chunk"; real rover, "TAMAGO-03 8-channel microphone array (36.5 mm radius, 16 kHz)". Not a VLA; no audio memory stated.
- **OmniEcho** (arXiv 2609.23407): FOA 4-channel 48 kHz benchmark from 30 real environments plus rendered training data; the model gets "the FOA spatial audio At within the current temporal window" and predicts K=8 steps; window length not found.
- **MuseVLA** (arXiv 2606.17598): microphone array (Sipeed 6+1) beamformed into a sound-intensity heatmap, "Acoustic-grounded object search: The robot localizes a hidden sound source using spatial audio cues"; real dexterous hand; window length not found.
- **FuSe** (arXiv 2501.04693): spectrogram through a ResNet26 into Octo and PaliGemma-based policies, real manipulation; window not found.
- **VLAS** (arXiv 2502.13508 v2): "padded to a fixed length of 3000 frames", "1500 hidden representations", frozen Whisper encoder, single spoken instruction.
- **Speech-pi0.5** (arXiv 2610.33866 v1, DAC 2026 late-breaking): "(N=1500 frames)" compressed with "K=32 learnable query vectors"; one-off instruction; LIBERO.
- **RoboOmni** (arXiv 2510.23763 v3, ICLR 2026): Appendix A.1: "The majority of clips range from 10 to 80 seconds. Overlapping dialogues tend to be shorter in duration, while non-verbal sequences are longer due to the insertion of sound events." Sec. 4.1: "At timestep t, given visual observation It and audio segment xt". The step-by-step audio handling at rollout time is not described in the passages read.
- **Audio-VLA** (arXiv 2511.09958 v1): "The raw audio stream is segmented into fixed-length chunks corresponding to each timestep"; two contact microphones at 44.1 kHz; length not stated.
- **ManiWAV** (arXiv 2406.19464 v2): Sec. 4: "The audio signal (from the last 2-3 seconds, depending on the task)"; Sec. 4.3 ablation: "[1s audio] yields significantly lower performance since the shortened audio window does not contain sufficient information ... [10s audio] shows that even though a longer audio window contains sufficient information, it increases the complexity of the learning process and requires a higher capacity model." Appendix B.2.1 pouring substeps (Grasp, Pour, Place): ours 90 / 100 / 90, 1 s audio 91.7 / 30 / 0, 10 s audio 100 / 60 / 16.7, vision only 100 / 0 / 8.3. This is the only audio-length ablation I found in an audio-conditioned policy.
- **Zhang and Gienger** (arXiv 2512.08405): flow-matching audio world model predicts future audio; "we incorporate the last L′ audio feature latents from the preceding window"; no window values found.
- **Audio-Visual World Models** (arXiv 2512.00883): "a 0.15-second binaural audio segment sampled at 16 kHz" per frame, "context length of 4 frames", lookahead planning for a navigation policy; a world model, not a VLA.
- **EchoVLA / "Listen, Look, Drive"** (arXiv 2601.12142): synthetic speech commands on nuScenes, open-loop; "sliding-window attention" in the streaming encoders. Abstract and keyword passes only.
- **VoiceGuard** (arXiv 2611.19984): an audio-native LLM routes spoken requests into a function packet; the VLA (SmolVLA) does not take audio (from the abstract). Not an audio-VLA by the table's definition.
- **HLR-LLM** (arXiv 2610.21588 v1): binaural spatial branch plus a monaural pathway in a question-answering model; no acting. Out of scope except as a spatial-audio encoder reference.
- Seen and not counted: HelixWorld (arXiv 2609.38123, stereo audio-visual world model for generation, abstract read), XBot-Omni (2611.17160) and MIRA (2609.24547) (human-robot interaction, speech out and in, no action policy with environmental audio), Cognitive Action Reasoning / ProAction (2609.23486, high-level action reasoning from audio, video and text; not opened beyond the abstract).

### (1) What "long-form" audio means in the audio literature

Plain idea: "long-form" is about whether a system can follow audio that lasts far longer than what it was trained and tested on (seconds), up to minutes or hours, without losing the thread. Sources opened (arXiv HTML, `verified`):

- **Long-form ASR.** Chiu et al. 2019 (arXiv 1911.02242 v1): "previous studies have focused primarily on short utterances that typically last for just a few seconds or, at most, a few tens of seconds. Whether such architectures are practical on long utterances that last from minutes to hours remains an open question." Whisper (arXiv 2212.04356), Sec. 3.8: "Whisper models are trained on 30-second audio chunks and cannot consume longer audio inputs at once" and long audio is handled by "buffered transcription ... by consecutively transcribing 30-second segments of audio and shifting the window according to the timestamps predicted by the model"; Fig. 6 compares ASR "where the input lengths range from a few minutes to a few hours". Murmur (arXiv 2606.01483): "an intermediate chunk size of about five minutes matches single-pass accuracy while avoiding repetition loops". Omnilingual-MsASR (arXiv 2610.29455): "recordings up to 30 minutes".
- **Long audio understanding benchmarks.** BLAB (arXiv 2505.03054): audio language models are "typically under 30 seconds"; BLAB uses "audio segments averaging 51 minutes in length" and "833+ hours"; all six models tested "struggle". AudioMarathon (arXiv 2510.07293): "durations ranging from 90.0 to 300.0 seconds"; "clear performance drops as audio length grows". LongAudioSpan (arXiv 2608.26431): "from 10 minutes to over 2 hours". VoiceGiraffe (arXiv 2605.27976): "understanding hour-level audio remains a fundamental bottleneck". Logbook (arXiv 2610.07338): "recordings ranging from ten minutes to six days"; "Even modern audio language models with longer context are evaluated on at most five minutes"; end-to-end systems are often better than cascaded ones "but degrades with longer context".
- **Streaming and memory for long audio.** OmniMem (arXiv 2606.07577): a streaming audio-visual language model limited by "the linear growth of video tokens and key-value (KV) caches", with a memory budget; evaluated on long video benchmarks (abstract and introduction read).

So in this literature "long-form" is minutes (about 5 to 30) to hours, occasionally days; "short" is under about 30 seconds; the recurring problems are attention cost, losing information across chunks, and temporal grounding over long spans. Nothing here is about robots. The audio-VLAs above take 0.5 to 10 s at a step. HEAR's memory is the nearest thing to long listening, but no memory span in minutes is stated in the text I read. `verified` for the sources; the comparison is mine.

### (2) What none of the audio-VLAs does (a search result, labelled)

Search result (`verified` for each paper's own statements; "not found" means not found in the passages I read and the searches below, not proof of absence):
1. **Long-form audio processing.** No audio-VLA takes minutes-to-hours audio at inference. The longest per-input audio found is RoboOmni's dialogue clips (10 to 80 s), used as an instruction. HEAR's own limitation framing is about milliseconds-to-seconds events across a decision gap, not minutes.
2. **Audio length as a controlled variable.** Only ManiWAV's 1 s / 2-3 s / 10 s ablation (longer did worse). None in a VLA, none in navigation.
3. **Persistent audio memory with a measured span.** HEAR has the module but the span is not stated; LH-AVLN's memory is a map of cues.
4. **Spatial audio inside a world model that also predicts video and actions.** Audio-Visual World Models (navigation) predicts audio and video but is not a VLA; no audio-VLA predicts video.
5. **Spatial audio in a real-robot VLA.** SoundAct and S2A2 (real, ACT) are not VLAs; the VLA-style spatial-audio work (NaVLA2, ADAPT, LH-AVLN) is simulation only.
6. **Controlled mono, stereo and none with the encoder fixed in a VLA.** NaVLA2's Table 2 changes the encoder with the channel count (CLAP versus SpatialAST); no seeds stated (`recalled` from the survey for the encoder names CLAP and SpatialAST and the Table 2 numbers; the PDF text and the released code do show two audio branches, a mono-downmix one and a two-channel one, with different preprocessing).
7. **Moving sound or changing source over time with an audio history** in VLAs: LH-AVLN alternates sources, but only for a training-free agent.

### Speculative: gaps that would suit a spatial-audio world action model

`speculative` (my reading, not a finding):
- **Audio history over an episode (minutes, not seconds).** A world model already carries a state over time, so a spatial-audio state that keeps "where was that sound" after it stops looks natural; HEAR's memory is mono and manipulation-only, LH-AVLN's is a map and not learned. An experiment: vary how long the audio history is (a few seconds against the whole episode) with the same model and measure navigation success, with ManiWAV's 10 s result as the warning that longer is not automatically better.
- **Sound that stops before the agent arrives.** SoundSpaces-style navigation keeps the source playing; a one-off or intermittent sound tests memory, which none of the navigation audio-VLAs does.
- **Predicting audio and video together under spatial audio** (the world-model role), which the VLAs above skip.
- A caution: "long-form" in the audio literature means minutes to hours of continuous audio, and a navigation episode in SoundSpaces is on the order of tens of steps. Whether episodes in WNM-3D's simulator are long enough to need "long-form" memory is not known to me; the PI's meaning of "long-form" was not confirmed.

### Not verified, or open
- Which of the "not found" window lengths (SoundAct, OmniEcho, FuSe, MuseVLA, Audio-VLA, RoboOmni at step level, BCNav) are in the PDFs or code but not in the text I extracted. NaVLA2's window was only found in the released code; whether the trained model used the same values is not checked.
- OpenReview papers (Hear-to-Act, HearVGo, D2AVN, TraSNav, SoundAct PDF) and the CVPR "Hear You Are" paper: access refused (HTTP 403), not opened.
- Whether newer arXiv papers (after about 2026-11-17) exist; the arXiv API returned HTTP 429 and 503 for my keyword queries, so discovery used the arXiv website search pages instead.

### Searches run (2026-11-19)
- arXiv search pages (https://arxiv.org/search/, all fields, newest first, first 50 results): "audio vision-language-action"; "sound robot manipulation policy VLA" (no results returned); "speech instruction vision-language-action"; "audio-visual navigation vision-language model binaural" (no results returned); "microphone array robot policy imitation learning" (no results returned); "audio world action model"; "streaming audio robot"; "spatial audio embodied agent navigation"; "long-form audio understanding benchmark"; "long audio understanding large audio language model minutes hours"; "long-form speech recognition"; "AudioMarathon long-context audio".
- arXiv API (export.arxiv.org) keyword queries on audio, sound, speech, binaural and VLA: HTTP 429 or 503; one query ("omni-modal" and "vision-language-action") returned OmniVLA and RoboOmni entries only (OmniVLA entries not opened).
- Citation and name leads taken from: the 2026-11-16 survey files (audio-in-robot-policies, spatial-audio-in-embodied-ai, prior-work-check) and the 2026-11-18 novelty file (ListenVLA, ADAPT, BCNav, HearVGo, Hear-to-Act, SoundAct).
- Direct opens: arXiv HTML for the ids named above; `tanmingkui.github.io/files/publications/NaVLA2.pdf`; `gh api repos/felixfjg/NaVLA/tarball` (commit 436ad6b); `vision3d-lab.github.io/soundact/`; OpenReview PDF `yakI3P3Ib6` (HTTP 403, not retried).
- Names from the task brief not found as separate audio-VLA papers by title in these searches: "Audio-WM" (the nearest is Zhang and Gienger, 2512.08405), "Hear-to-Act" and "SoundAct" (OpenReview only), "S2A2" and "ADAPT" (found). No separate arXiv paper named "ListenVLA" other than 2610.16313.

Local copies (untrusted, never executed): `the scratch directory `.
