# Changelog

**1.5** (2026-11-13)
- new map model: sideways swings and dots like mappers make them, fewer crossed doubles
- new rhythm model: busy parts follow the song, fast drops stay fast at high stars
- new lighting model: lights come up on the notes, left and right take turns, patterns repeat like a mapper's
- lights made about twice as fast
- the beat grid follows the song, so fewer missing or late swings
- more doubles, like human maps
- stars come more from the rhythm, less from extra-hard patterns
- maps stop when the song fades out
- the Auto scene picks every scene equally often
- replays: chain links get cut

**1.4** (2026-11-08)
- new map model with chains, and Rare / Normal / Lots for arcs and chains
- new lighting model, hears the whole song at once
- arcs shaped more like mappers make them
- more walls, closer to ranked maps
- tech and speed styles lean further into their style
- songs that can be counted at double time: the fast count for speed and Expert+, the slow one for tech
- fewer awkward resets: no floor bombs right before an up swing
- maps save as v3 by default, since v2 can't hold chains
- YouTube links: the video's thumbnail as the cover when no artwork is found, and a track number in the title isn't taken for the artist

**1.3.1** (2026-11-07)
- reset bombs sit where the hand is, not off to the side
- Stop works while the song is being read and the rhythm made

**1.3** (2026-11-06)
- new map model, far fewer odd resets
- new rhythm model
- lights follow the song more, calmer when it's loud
- no more bomb walls
- replays: cleaner swings, fewer bad cuts

**1.2.4** (2026-11-03)
- no GPU: makes one try instead of three, so maps come much sooner but can land further under the stars asked
- the bar keeps moving through every try and Stop works the whole way
- says when there's no GPU, and shows when it's making the lights

**1.2.3** (2026-11-03)
- no more second cut the same way right behind the first
- resets with room get a bomb where the hand would be

**1.2.2** (2026-11-03)
- harder difficulties rate higher than easier ones

**1.2.1** (2026-11-03)
- reaction time and jump distance match ranked maps at each star level, on any song
- fewer sudden hard parts when a song builds up
- fewer odd patterns at high stars
- notes rarely hide the one after them
- fewer notes on the top row
- replays: fixed how notes are marked, for in-game replays

**1.2** (2026-11-01)
- new map model, trained from scratch on more songs
- a rhythm model picks the hits first
- stars closer to what you asked for
- lighting model, and a scene picker
- covers and artist names found online
- less repeating, no notes that make the sabers meet head on
- faster on Linux
- paste a YouTube link instead of adding a file
- acc style stays calmer at 8★ and up

**1.1** (2026-10-26)
- new map model, better at low and mid stars
- better BPM timing, songs that change tempo too
- replays: a player model plays your map
- small fixes: note speed per star level, no bomb floods, a progress bar

**1.0** (2026-10-25)
- first release
