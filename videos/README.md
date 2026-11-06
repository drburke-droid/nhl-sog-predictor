# Videos for social media

Every eye video and every speed-of-light video we finished, renamed so the file says what it is.
SHORT in a name means vertical 1080x1920 (Shorts, Reels, TikTok); everything else is 16:9 for YouTube.
Thumbnails are in each folder's `thumbnails/`; titles, descriptions and tags already written are in `copy/`.

## eye/

| File | Format | Length | What it is |
|---|---|---|---|
| vernier_you_can_see_smaller_than_your_pixels_6m19s_4K.mp4 | 16:9, 4K | 6:19 | Vernier hyperacuity (replaces the older hyperacuity video). Smaller 51 MB copy; the 150 MB 4K master was sent separately. Copy: `copy/vernier_YOUTUBE.md`, thumbnails vernier_thumb_A/B/C |
| glasses_work_by_slowing_light_SHORT_68s.mp4 | vertical | 1:08 | Your glasses work by slowing light down (lenses, refraction, thin vs thick lens, the Mariana Trench) |
| blur_adaptation_7m09s.mp4 | 16:9 | 7:09 | Blur adaptation. Copy: `copy/blur_adaptation_YOUTUBE.md`, thumbnails blur_adaptation_thumb_A/B/C |
| blur_adaptation_we_didnt_start_the_fire_edition_7m16s.mp4 | 16:9 | 7:16 | Same film, "We Didn't Start the Fire" edition (pick one of the two). Copy: `copy/blur_adaptation_fire_edition_YOUTUBE.md`, thumbnails blur_adaptation_fire_thumb_A/B/C |

Not here, because neither was saved to any repository: the BVD video (binocular vision dysfunction; AI-voice and
no-voice versions plus Shorts, made with the neurogaze project) and the video of glasses lenses growing thicker up to -39D.

## eye/: the AE/AF lock trick (in progress, waiting on the voiceover)

The phone AE/AF lock trick from the 2023-11-28 blog post, filmed with a −3.00 lens.

| File | Format | Length | What it is |
|---|---|---|---|
| aeaf_lock_trick_1m44s.mp4 | 16:9, 1080p | 1:44 | The whole video with music and sound effects, no voice yet |
| audio/aeaf_lock_music.wav | 48 kHz stereo | 1:44 | Music stem (original, synthesized for this video; -17 LUFS) |
| audio/aeaf_lock_sfx.wav | 48 kHz stereo | 1:44 | Sound-effects stem (card whooshes, lock tick, shutter, wipe) |
| aeaf_lock_trick_SHORT_56s.mp4 | vertical 1080x1920 | 0:56 | Shorts cut: hook, the five steps (quicker), with vs. without stacked, why it works, outro |
| audio/aeaf_lock_SHORT_music.wav, audio/aeaf_lock_SHORT_sfx.wav | 48 kHz stereo | 0:56 | Music and sound-effects stems for the Short |

Voiceover script, timed to both cuts: `copy/aeaf_lock_VO_script.md`. Titles, description, chapters and tags: `copy/aeaf_lock_YOUTUBE.md`;
thumbnails `thumbnails/aeaf_lock_thumb_A/B/C.png` and a Shorts cover `thumbnails/aeaf_lock_short_cover.png`. To add the voice, put the video's picture,
the two stems and the voice on separate tracks and lower the music under the voice.

What happens when: 0:00 intro ("Make your phone camera depend on your glasses"); 0:10.5 the demo, which pauses on
five step cards (0:10.5, 0:16.5, 0:27, 0:35.5, 0:43), plays slowed down between them, holds on FOCUS LOCKED at 0:32.5 and
ends tight on the blurry screen (photo taken at 0:47); 0:52.5 "How a −3.00 nearsighted person sees" side by side, 0:59.5
a wipe between the photos; 1:07 why it works; 1:18 up close clear, far away blurry; 1:28 the catch (no astigmatism);
1:37.5 "Show your friends how you really see."
Build scripts, the phone track and the two photos are in `tools/aeaf_lock/` (`make_aeaf.py` for the picture,
`make_short.py` for the Short,
`music.py` for the music and effects, `thumbs.py` and `short_cover.py` for the images).

## speed_of_light/

| File | Format | Length | What it is |
|---|---|---|---|
| all_world_roads_at_light_speed_4m40s.mp4 | 16:9 | 4:40 | The full film: light racing along the world's roads |
| world_roads_light_speed_SHORT_world.mp4 | vertical | 0:47 | Short cut: the world |
| world_roads_light_speed_SHORT_americas.mp4 | vertical | 0:53 | Short cut: the Americas |
| world_roads_light_speed_SHORT_finale.mp4 | vertical | 0:33 | Short cut: the finale |
| every_flight_on_earth_at_light_speed_11m02s.mp4 | 16:9 | 11:02 | Every flight on Earth (YouTube upload version, with music). Thumbnails every_flight_thumbnail_v2A/B |
| every_flight_on_earth_SHORT.mp4 | vertical | 0:37 | Every flight, short cut |
| calgary_every_street_at_light_speed.mp4 | vertical | 0:17 | Light through every street in Calgary |
| blink_vs_calgary_at_light_speed.mp4 | 3:2 (1200x800) | 0:16 | A blink against light crossing Calgary; square-ish, suits a feed post |
| lost_photon_LA_to_NY_coin_flips_SHORT.mp4 | vertical | 0:25 | A lost photon finding its way from LA to New York by coin flips |
| light_speedrun_LA_to_NY_SHORT.mp4 | vertical | 0:52 | "Light speedrun", LA to New York |
| 48_cities_at_light_speed_SHORT.mp4 | vertical | 1:03 | Race of light across 48 cities |
| roomba_at_light_speed_every_road.mp4 | 16:9 | 3:23 | A Roomba-style photon covering every road. Copy: `copy/roomba_YOUTUBE.md`, thumbnails roomba_thumb_A/B/C |
| roomba_at_light_speed_SHORT.mp4 | vertical | 0:50 | Roomba photon, short cut |
| roomba_at_light_speed_calgary_SHORT.mp4 | vertical | 0:42 | Roomba photon in Calgary |

Left out as superseded or drafts: the first hyperacuity video, the first cut of the world short, the world-roads
preview, the daytime-only half of the flights film, and the signal-fires film (not speed of light).
