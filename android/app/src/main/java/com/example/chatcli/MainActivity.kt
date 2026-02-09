package com.example.chatcli

import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject

class MainActivity : AppCompatActivity() {
    private val client = OkHttpClient()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val scrollView = findViewById<ScrollView>(R.id.conversation_scroll)
        val conversationView = findViewById<TextView>(R.id.conversation_text)
        val inputField = findViewById<EditText>(R.id.prompt_input)
        val sendButton = findViewById<Button>(R.id.send_button)

        fun appendLine(text: String) {
            conversationView.append(text)
            conversationView.append("\n\n")
            scrollView.post { scrollView.fullScroll(View.FOCUS_DOWN) }
        }

        sendButton.setOnClickListener {
            val userText = inputField.text.toString().trim()
            if (userText.isEmpty()) return@setOnClickListener

            if (BuildConfig.OPENAI_API_KEY.isBlank()) {
                appendLine("System: Missing OPENAI_API_KEY in local.properties.")
                return@setOnClickListener
            }

            inputField.setText("")
            appendLine("You: $userText")
            sendButton.isEnabled = false

            lifecycleScope.launch {
                val reply = withContext(Dispatchers.IO) {
                    runCatching { fetchResponse(userText) }
                        .getOrElse { "Error: ${it.localizedMessage}" }
                }
                appendLine("Assistant: $reply")
                sendButton.isEnabled = true
            }
        }
    }

    private fun fetchResponse(prompt: String): String {
        val payload = JSONObject()
            .put("model", "gpt-5.2")
            .put("input", prompt)

        val request = Request.Builder()
            .url("https://api.openai.com/v1/responses")
            .addHeader("Authorization", "Bearer ${BuildConfig.OPENAI_API_KEY}")
            .addHeader("Content-Type", "application/json")
            .post(payload.toString().toRequestBody("application/json".toMediaType()))
            .build()

        client.newCall(request).execute().use { response ->
            val body = response.body?.string().orEmpty()
            if (!response.isSuccessful) {
                return parseError(body) ?: "Request failed with HTTP ${response.code}"
            }
            return extractOutputText(body)
        }
    }

    private fun parseError(body: String): String? {
        return runCatching {
            val error = JSONObject(body).optJSONObject("error") ?: return@runCatching null
            error.optString("message").takeIf { it.isNotBlank() }
        }.getOrNull()
    }

    private fun extractOutputText(body: String): String {
        return runCatching {
            val root = JSONObject(body)
            val direct = root.optString("output_text")
            if (direct.isNotBlank()) return@runCatching direct

            val output = root.optJSONArray("output") ?: return@runCatching body
            val chunks = mutableListOf<String>()
            for (i in 0 until output.length()) {
                val item = output.optJSONObject(i) ?: continue
                if (item.optString("type") != "message") continue
                val content = item.optJSONArray("content") ?: JSONArray()
                for (j in 0 until content.length()) {
                    val part = content.optJSONObject(j) ?: continue
                    if (part.optString("type") == "output_text") {
                        val text = part.optString("text")
                        if (text.isNotBlank()) chunks.add(text)
                    }
                }
            }
            if (chunks.isEmpty()) body else chunks.joinToString(separator = "\n")
        }.getOrDefault(body)
    }
}
