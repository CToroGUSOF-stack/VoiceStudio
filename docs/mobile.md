# VoiceStudio on your phone (iOS and Android)

The iOS and Android apps run VoiceStudio **on your phone**: speech, voice
cloning, dictation and dubbing work offline once their models are downloaded,
with no computer, account or cloud service. If you also run VoiceStudio on a
computer (desktop app, Docker or a server), you can pair the phone with it and
choose, per task, to run on **This phone** or **My computer**.

> **Preview.** The apps aren't in the App Store or Google Play yet. Build them
> from source (below).

| | iOS | Android |
|---|---|---|
| Minimum OS | iOS 17 (iPhone, iPad) | Android 10 (API 29) |
| Source | [`mobile/ios/`](../mobile/ios/README.md) | [`mobile/android/`](../mobile/android/README.md) |

## What you can do

- **Speak**: type text, pick a voice and language, generate, play, share.
- **Voices**: clone a voice from a recording or an audio file. Voices you
  clone on the phone stay on the phone.
- **Dub**: pick a video or audio file and target languages; the phone
  transcribes, translates and re-voices it in the original speaker's voice.
- **Dictate**: tap the mic and watch the transcript appear live.
- **Models**: download, update or delete the on-device models.

## On-device models

Open **Models** and download what you need. Nothing downloads until you tap
it; files come from Hugging Face and are checked before use. Wi-Fi only unless
you allow mobile data.

| Model | For | Size | Phone memory |
|---|---|---|---|
| Kitten (English) / Kokoro (multi-language) | Speak | ~170 MB each | 2–3 GB |
| Whisper tiny / base | Transcription, dubbing | 100–160 MB | 2–3 GB |
| Zipformer | Live dictation (English) | 44 MB | 2 GB |
| OmniVoice | Voice cloning, dubbing | 660 MB | 6 GB |
| Qwen3 1.7B | Dub translation | 1.1 GB | 6 GB |

Some models ask you to accept their licence first (for example OmniVoice is
for non-commercial use), the same as on the desktop. Models your phone can't
run are shown greyed out with the reason.

## Pair a computer (optional)

1. On the computer, open VoiceStudio, click the **Local** pill in the footer
   and choose **Network** ([how sharing works](sharing.md)). On a different network, use
   [Tailscale](sharing.md) or an HTTPS address instead.
2. In the app, tap **Scan QR code** and scan the code the panel shows. Or tap
   **Enter address** and type the address and the 6-digit PIN.
3. For a server protected by an [API key](api-auth.md), enter the key once.
   The app swaps it for a session and never stores the key; **Disconnect**
   ends the session.

Plain `http://` works only for addresses on your own network (home Wi-Fi,
`.local`, Tailscale); everything else must be `https://`.

## If something doesn't work

- **"Accept the licence on your computer"**: some models need their licence
  accepted first. Open VoiceStudio on the computer → **Model Catalogue** and
  accept it there; the phone can't accept licences.
- **Engine shown as unavailable**: install it on the computer. The reason the
  app shows comes from the computer.
- **Dictate isn't ready**: install the speech-to-text model the app names, on
  the computer. Remote dictation also needs an API-key session or a trusted
  network set up on the computer; the share PIN alone isn't enough.
- **Can't connect**: make sure the phone and computer are on the same network
  (or both on Tailscale), sharing is on, and you used the IP address the panel
  shows. See [troubleshooting](install/troubleshooting.md).

## For developers

Both apps follow one contract, [`mobile/contract/SPEC.md`](../mobile/contract/SPEC.md):
a feature ships on both or neither. Colors, the API description and the text
are generated from the desktop app and backend, never copied by hand:

```sh
bun scripts/export-mobile-tokens.ts        # theme tokens from the Electron theme
uv run python scripts/export_openapi.py    # mobile/contract/openapi.json from the backend
```

`tests/test_mobile_contract.py` fails when either is stale.

**Try the iPhone app without Xcode.** Each `mobile` workflow run uploads an
unsigned build, `VoiceStudio-ios-unsigned` (kept 7 days). Download it from the
run's Artifacts, then install it with a sideloading tool that signs it with your
own Apple ID (for example Sideloadly). With a free Apple ID the install expires
after 7 days; turn on Developer Mode on the iPhone (Settings → Privacy &
Security) when iOS asks. App text uses the
Electron locale catalogs (`mobile.*` keys). The `mobile` workflow builds and
tests both apps.
