# chat-cli

This repo now includes an Android app that mirrors the CLI experience. The Android project lives under `android/` and uses the OpenAI Responses API to send prompts and render replies.

## Android quick start

1. Open `android/` in Android Studio (or use the Gradle wrapper).
2. Set your API key in `android/local.properties`:

```
OPENAI_API_KEY=your_key_here
```

3. Build and run the `app` configuration on an emulator or device, or run:

```
cd android
./gradlew :app:assembleDebug
```

> Note: The API key is read at build time and exposed to the app as a `BuildConfig` field for simplicity. For production, move this to a secure backend.

## CLI

The original Python CLI is still available via `python main.py`.
