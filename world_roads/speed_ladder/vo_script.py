"""What Lily says (Kokoro-82M, voice bf_lily, vo_tts.py), line by line: (spoken, shown). The spoken words spell out
numbers and names the way they are said; the shown words are the subtitles. She is silent whenever anything flies at
real speed (part 1, the Wright Flyer in part 2, the races' first seconds), so those keep their own sound."""

LINES = {
    # the opening
    "o1": ("How fast is light?", "How fast is light?"),
    "o2": ("It's so fast, it's hard to imagine. So let's work up to it, with a hundred and twenty years of the fastest "
           "machines ever built, flying past one camera, on a desert road.",
           "It's so fast, it's hard to imagine. So let's work up to it: 120 years of the fastest machines ever built, "
           "flying past one camera on a desert road."),
    "c1": ("Part one. Real speed. This is exactly what an ordinary camera sees: twenty-five pictures every second. "
           "No tricks. Just watch.",
           "Part 1: real speed. Exactly what an ordinary camera sees, 25 pictures a second. No tricks. Just watch."),
    "p1e": ("That flash was the Parker Solar Probe, the fastest machine ever built. It crossed in a single frame, and "
            "nothing in the picture survived its shock wave. And yet, light is still more than fifteen hundred times "
            "faster.",
            "That flash was the Parker Solar Probe, the fastest machine ever built. It crossed in a single frame, and "
            "nothing in the picture survived its shock wave. Yet light is still more than 1,500 times faster."),
    # part 2
    "c2": ("Part two. The same planes, slowed down forty-eight times, like a camera filming twelve hundred pictures a "
           "second. Only the Wright Flyer stays at real speed.",
           "Part 2: the same planes, slowed down 48 times, like a camera filming 1,200 pictures a second. Only the "
           "Wright Flyer stays at real speed."),
    "p2a": ("The Spitfire, from the Second World War. Three hundred and seventy miles an hour. Even slowed down this "
            "much, it covers a mile in about ten seconds. Light covers a hundred and eighty-six thousand miles, every "
            "second.",
            "The Spitfire, from the Second World War: 370 mph. Even slowed down this much, it covers a mile in about "
            "10 seconds. Light covers 186,282 miles every second."),
    "p2b": ("The Bell X one, the first plane to fly faster than sound. Its shock wave whips up the desert dust behind "
            "it.",
            "The Bell X-1, the first plane to fly faster than sound. Its shock wave whips up the desert dust behind "
            "it."),
    "p2c": ("The F thirty-five, at one and a half times the speed of sound. A mile every three seconds.",
            "The F-35, at 1½ times the speed of sound: a mile every 3 seconds."),
    "p2d": ("And the X fifteen, still the fastest plane ever flown by a pilot. More than a mile, every second.",
            "And the X-15, still the fastest plane ever flown by a pilot: more than a mile every second."),
    "p2e": ("Yet light is a hundred and forty-eight thousand times faster still.",
            "Yet light is 148,000 times faster still."),
    # part 3
    "c3": ("Part three. A race. Each machine sets off at just the right moment, so that all five reach the right-hand "
           "edge at exactly the same instant.",
           "Part 3: a race. Each machine sets off at just the right moment, so all five reach the right-hand edge at "
           "exactly the same instant."),
    "p3a": ("To see the planes, we slow time down forty-eight times. The Wright Flyer set off first.",
            "To see the planes, we slow time down 48 times. The Wright Flyer set off first."),
    "p3b": ("Then the F thirty-five, and the X fifteen, each one later, and faster.",
            "Then the F-35, and the X-15, each one later and faster."),
    "p3c": ("Here comes Parker. It's so fast, we slow time again, a thousand times.",
            "Here comes Parker. It's so fast, we slow time again: 1,000 times."),
    "p3d": ("All five cross the line in the same twenty-five-thousandth of a second.",
            "All five cross the line in the same 1/25,000 of a second."),
    # light
    "e1": ("Parker is the fastest machine we've ever built. But light is in a completely different league.",
           "Parker is the fastest machine we've ever built. But light is in a completely different league."),
    "e2": ("Six hundred and seventy million miles an hour. That's a hundred and eighty-six thousand miles, every single "
           "second.",
           "670 million miles an hour. That's 186,282 miles every single second."),
    "e3": ("Here are their speeds, drawn to scale.", "Here are their speeds, drawn to scale."),
    "e4": ("Next to Parker, the planes barely show up at all.", "Next to Parker, the planes barely show up at all."),
    "e5": ("Now, let's add light.", "Now, let's add light."),
    "e6": ("Parker's bar shrinks to less than a single pixel. Light is fifteen hundred and sixty times faster than "
           "Parker.",
           "Parker's bar shrinks to less than a single pixel. Light is 1,560 times faster than Parker."),
    "e7": ("In just one second, light could travel all the way round the Earth, seven and a half times.",
           "In just one second, light could travel all the way round the Earth 7½ times."),
    "e8": ("One lap takes about an eighth of a second.", "One lap takes about an eighth of a second."),
    "e9": ("Light really travels in straight lines, so imagine it being steered round the equator.",
           "Light really travels in straight lines, so imagine it being steered round the equator."),
    "e10": ("So, in part four, we'll run the same race again, with a pulse of light.",
            "So in part 4, we'll run the same race again, with a pulse of light."),
    "e11": ("It sets off the moment the Wright Flyer appears, and races round the Earth.",
            "It sets off the moment the Wright Flyer appears, and races round the Earth."),
    "e12": ("Every time it comes back round, it passes our road again, and leaves a faint line in the sky, a little "
            "higher than the last.",
            "Every time it comes back round, it passes our road again and leaves a faint line in the sky, a little "
            "higher than the last."),
    "e13": ("Real light leaves no trail. The lines are just markers, so you can count the laps as they pile up.",
            "Real light leaves no trail. The lines are just markers, so you can count the laps as they pile up."),
    "e14": ("Here's how time will run.", "Here's how time will run."),
    "e15": ("Light is so fast that even slowed down a thousand times, it would cross the picture between two frames.",
            "Light is so fast that even slowed down 1,000 times, it would cross the picture between two frames."),
    "e16": ("So twice, you'll see the words, time stopped. Time doesn't really stop. We slow it down a million times, "
            "like a camera filming twenty-five million pictures a second, just long enough to watch light go by.",
            "So twice, you'll see the words TIME STOPPED. Time doesn't really stop: we slow it down a million times, "
            "like a camera filming 25 million pictures a second, just long enough to watch light go by."),
    "e17": ("Then the race carries on, from the exact same moment.", "Then the race carries on from the exact same moment."),
    "c4": ("Part four. The race, with light. Watch the sky.", "Part 4: the race, with light. Watch the sky."),
    # part 4
    "f1": ("Time stopped. Well, not really. We've slowed it down a million times, so even light looks slow. Watch the "
           "horizon, on the left.",
           "TIME STOPPED. Well, not really: we've slowed it down a million times, so even light looks slow. Watch the "
           "horizon, on the left."),
    "f2": ("There it goes. It crosses the whole picture in about a millionth of a second.",
           "There it goes. It crosses the whole picture in about a millionth of a second."),
    "f3": ("Now, back to real speed. Light is off round the Earth.",
           "Now, back to real speed. Light is off round the Earth."),
    "f4": ("Each new line is another lap of the Earth. In under two seconds, light has been round more than a dozen "
           "times, and the Wright Flyer still hasn't crossed the road.",
           "Each new line is another lap of the Earth. In under 2 seconds, light has been round more than a dozen "
           "times, and the Wright Flyer still hasn't crossed the road."),
    "f5": ("Slowed forty-eight times, a lap takes six seconds now.",
           "Slowed 48 times, a lap takes 6 seconds now."),
    "f5b": ("Here comes Parker.", "Here comes Parker."),
    "f6": ("Time stops again, a million times slower. Parker is frozen in its tracks, but light isn't. Watch the top of "
           "the ladder.",
           "TIME STOPPED again: a million times slower. Parker is frozen in its tracks, but light isn't. Watch the top "
           "of the ladder."),
    "f7": ("There goes light, on its seventeenth lap of the Earth, racing past Parker as if it were standing still.",
           "There goes light, on its 17th lap of the Earth, racing past Parker as if it were standing still."),
    "f8": ("Back to a thousand times slower. Parker carries on to the finish.",
           "Back to 1,000 times slower. Parker carries on to the finish."),
    "f9": ("All five finish together. And even Parker is fifteen hundred and sixty times slower than light.",
           "All five finish together. And even Parker is 1,560 times slower than light."),
    "f10": ("While the Wright Flyer crossed one road, light went round the entire world, seventeen times.",
            "While the Wright Flyer crossed one road, light went round the entire world 17 times."),
    # the ending
    "m1": ("How long would each of them take, to fly to the Moon?", "How long would each of them take to fly to the Moon?"),
    "m2": ("The Wright Flyer, almost a year. The Spitfire, four weeks. The X fifteen, two days. Parker, about half an "
           "hour.",
           "The Wright Flyer: almost a year. The Spitfire: 4 weeks. The X-15: 2 days. Parker: about half an hour."),
    "m3": ("And light? Just over one second.", "And light? Just over one second."),
    "m4": ("Watch. This is light, crossing to the Moon, at its real speed.",
           "Watch: this is light crossing to the Moon, at its real speed."),
    "m5": ("Light. Six hundred and seventy million miles an hour. Nothing with mass can ever catch it.",
           "Light: 670 million miles an hour. Nothing with mass can ever catch it."),
}


if __name__ == "__main__":
    import json, sys
    json.dump({k: v[0] for k, v in LINES.items()}, open(sys.argv[1], "w"), indent=1)
