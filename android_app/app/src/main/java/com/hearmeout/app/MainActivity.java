package com.hearmeout.app;

import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.speech.tts.TextToSpeech;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.Space;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends Activity {
    private static final int NAVY = Color.rgb(6, 31, 54);
    private static final int PANEL = Color.rgb(247, 251, 255);
    private static final int BLUE = Color.rgb(19, 184, 244);
    private static final int BLUE_DARK = Color.rgb(26, 102, 232);
    private static final int REQUEST_RECORD = 7001;
    private static final int REQUEST_UPLOAD = 7002;

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final Handler main = new Handler(Looper.getMainLooper());
    private SharedPreferences prefs;
    private TextToSpeech tts;
    private String token = "";
    private JSONObject user;
    private JSONObject lastResult;
    private int introIndex = 0;

    private final String[][] introSlides = new String[][]{
            {"HearMeOut", "AI-powered lip reading assistant"},
            {"Read Lips with AI", "Transform lip movements into accurate text using advanced AI technology."},
            {"Upload or Record", "Upload a video or record live to start real-time lip reading."},
            {"Get Instant Results", "Receive fast and clear text predictions within seconds."}
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        prefs = getSharedPreferences("hearmeout", MODE_PRIVATE);
        token = prefs.getString("token", "");
        tts = new TextToSpeech(this, status -> {
            if (status == TextToSpeech.SUCCESS) tts.setLanguage(Locale.US);
        });
        if (token.isEmpty()) {
            showStart();
        } else {
            call("GET", "/api/me", null, null, result -> {
                user = result.getJSONObject("user");
                showHome();
            }, error -> {
                prefs.edit().remove("token").apply();
                token = "";
                showStart();
            });
        }
    }

    @Override
    protected void onDestroy() {
        executor.shutdownNow();
        if (tts != null) {
            tts.stop();
            tts.shutdown();
        }
        super.onDestroy();
    }

    private String serverUrl() {
        String value = prefs.getString("server_url", "http://192.168.1.20:8000");
        if (value.endsWith("/")) return value.substring(0, value.length() - 1);
        return value;
    }

    private void saveServerUrl(String value) {
        prefs.edit().putString("server_url", value.trim()).apply();
    }

    private TextView text(String value, int sp, int color, int style) {
        TextView view = new TextView(this);
        view.setText(value);
        view.setTextSize(sp);
        view.setTextColor(color);
        view.setTypeface(null, style);
        view.setLineSpacing(0, 1.12f);
        return view;
    }

    private Button button(String label, int bg, int fg) {
        Button b = new Button(this);
        b.setText(label);
        b.setAllCaps(false);
        b.setTextColor(fg);
        b.setTextSize(15);
        b.setTypeface(null, 1);
        b.setBackgroundColor(bg);
        b.setMinHeight(dp(48));
        return b;
    }

    private EditText input(String hint) {
        EditText e = new EditText(this);
        e.setHint(hint);
        e.setHintTextColor(Color.rgb(155, 174, 190));
        e.setTextColor(Color.WHITE);
        e.setSingleLine(true);
        e.setPadding(dp(12), 0, dp(12), 0);
        e.setBackgroundColor(Color.argb(34, 255, 255, 255));
        return e;
    }

    private LinearLayout column(int bg, boolean padded) {
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setBackgroundColor(bg);
        if (padded) layout.setPadding(dp(18), dp(26), dp(18), dp(18));
        return layout;
    }

    private void add(ViewGroup parent, View child) {
        parent.addView(child, new ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        ));
    }

    private void gap(ViewGroup parent, int height) {
        Space space = new Space(this);
        parent.addView(space, new ViewGroup.LayoutParams(1, dp(height)));
    }

    private int dp(int value) {
        return (int) (value * getResources().getDisplayMetrics().density + 0.5f);
    }

    private void setScreen(View view) {
        setContentView(view);
    }

    private void showStart() {
        LinearLayout root = column(NAVY, true);
        root.setGravity(Gravity.CENTER);
        ImageView logo = new ImageView(this);
        logo.setImageResource(com.hearmeout.app.R.drawable.ic_launcher_foreground);
        LinearLayout.LayoutParams logoParams = new LinearLayout.LayoutParams(dp(190), dp(190));
        logoParams.gravity = Gravity.CENTER_HORIZONTAL;
        root.addView(logo, logoParams);
        gap(root, 12);
        add(root, centeredText(introSlides[introIndex][0], 28, Color.WHITE, 1));
        gap(root, 8);
        add(root, centeredText(introSlides[introIndex][1], 15, Color.rgb(214, 233, 247), 0));
        gap(root, 28);
        Button next = button(introIndex == 0 ? "Start" : "Next", BLUE, Color.WHITE);
        next.setOnClickListener(v -> {
            if (introIndex < introSlides.length - 1) {
                introIndex++;
                showStart();
            } else {
                showSignup();
            }
        });
        add(root, next);
        gap(root, 8);
        Button skip = button(introIndex < introSlides.length - 1 ? "Skip" : "Login", Color.TRANSPARENT, BLUE);
        skip.setOnClickListener(v -> {
            if (introIndex < introSlides.length - 1) {
                introIndex = introSlides.length - 1;
                showStart();
            } else {
                showLogin();
            }
        });
        add(root, skip);
        setScreen(root);
    }

    private TextView centeredText(String value, int sp, int color, int style) {
        TextView t = text(value, sp, color, style);
        t.setGravity(Gravity.CENTER);
        return t;
    }

    private void showSignup() {
        LinearLayout root = column(Color.rgb(49, 47, 67), true);
        gap(root, 68);
        add(root, text("Sign Up", 28, Color.WHITE, 1));
        add(root, text("Enter your details below and create a free account.", 14, Color.rgb(218, 230, 241), 0));
        gap(root, 16);
        EditText server = input("Laptop server URL");
        server.setText(serverUrl());
        EditText name = input("Name");
        EditText email = input("Your Email");
        EditText password = input("Password");
        password.setInputType(0x00000081);
        add(root, server);
        gap(root, 10);
        add(root, name);
        gap(root, 10);
        add(root, email);
        gap(root, 10);
        add(root, password);
        gap(root, 16);
        Button create = button("Create account", BLUE, Color.WHITE);
        create.setOnClickListener(v -> {
            saveServerUrl(server.getText().toString());
            JSONObject body = new JSONObject();
            try {
                body.put("display_name", name.getText().toString());
                body.put("email", email.getText().toString());
                body.put("password", password.getText().toString());
            } catch (Exception ignored) {
            }
            call("POST", "/api/signup", body, null, result -> acceptAuth(result), this::toast);
        });
        add(root, create);
        gap(root, 16);
        Button login = button("Already have an account? Log in", Color.TRANSPARENT, BLUE);
        login.setOnClickListener(v -> showLogin());
        add(root, login);
        setScreen(root);
    }

    private void showLogin() {
        LinearLayout root = column(Color.rgb(49, 47, 67), true);
        gap(root, 86);
        add(root, text("Log IN", 28, Color.WHITE, 1));
        add(root, text("Use the account saved on this laptop.", 14, Color.rgb(218, 230, 241), 0));
        gap(root, 16);
        EditText server = input("Laptop server URL");
        server.setText(serverUrl());
        EditText email = input("Your Email");
        EditText password = input("Password");
        password.setInputType(0x00000081);
        add(root, server);
        gap(root, 10);
        add(root, email);
        gap(root, 10);
        add(root, password);
        gap(root, 16);
        Button login = button("Log In", BLUE, Color.WHITE);
        login.setOnClickListener(v -> {
            saveServerUrl(server.getText().toString());
            JSONObject body = new JSONObject();
            try {
                body.put("email", email.getText().toString());
                body.put("password", password.getText().toString());
            } catch (Exception ignored) {
            }
            call("POST", "/api/login", body, null, result -> acceptAuth(result), this::toast);
        });
        add(root, login);
        gap(root, 16);
        Button signup = button("Do not have an account? Sign up", Color.TRANSPARENT, BLUE);
        signup.setOnClickListener(v -> showSignup());
        add(root, signup);
        setScreen(root);
    }

    private void acceptAuth(JSONObject result) {
        token = result.optString("token", "");
        user = result.optJSONObject("user");
        prefs.edit().putString("token", token).apply();
        showSuccess();
    }

    private void showSuccess() {
        LinearLayout root = column(NAVY, true);
        root.setGravity(Gravity.CENTER);
        LinearLayout card = column(Color.rgb(53, 52, 73), true);
        card.setGravity(Gravity.CENTER_HORIZONTAL);
        TextView check = centeredText("✓", 44, Color.WHITE, 1);
        add(card, check);
        add(card, centeredText("Success", 24, Color.WHITE, 1));
        add(card, centeredText("Congratulations, you have completed your registration.", 14, Color.rgb(218, 230, 241), 0));
        gap(card, 14);
        Button done = button("Done", BLUE, Color.WHITE);
        done.setOnClickListener(v -> showHome());
        add(card, done);
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        params.setMargins(dp(14), 0, dp(14), 0);
        root.addView(card, params);
        setScreen(root);
    }

    private String firstName() {
        String value = user == null ? "there" : user.optString("display_name", "there");
        return value.split(" ")[0];
    }

    private void showHome() {
        LinearLayout content = column(NAVY, true);
        add(content, text("Hi, " + firstName(), 24, Color.WHITE, 1));
        add(content, text("Analyze lip movements quickly and accurately using AI-powered speech recognition.", 14, Color.rgb(191, 211, 228), 0));
        gap(content, 16);
        add(content, action("Record Live", "Capture live lip movements instantly.", v -> showLive()));
        gap(content, 12);
        add(content, action("Upload Video", "Analyze speech from uploaded videos.", v -> openUploadPicker()));
        gap(content, 12);
        add(content, action("History", "View your saved analyses and results.", v -> showHistory()));
        gap(content, 18);
        LinearLayout tips = column(Color.rgb(10, 93, 146), true);
        add(tips, text("Tips for Better Accuracy", 17, Color.WHITE, 1));
        add(tips, text("• Use good lighting\n• Keep your face centered\n• Avoid excessive head movement\n• Speak clearly and slowly", 14, Color.WHITE, 0));
        add(content, tips);
        setScreen(withBottomNav(content, "home"));
    }

    private View action(String title, String sub, View.OnClickListener listener) {
        LinearLayout card = column(BLUE_DARK, true);
        card.setOnClickListener(listener);
        add(card, text(title, 18, Color.WHITE, 1));
        add(card, text(sub, 13, Color.rgb(232, 248, 255), 0));
        return card;
    }

    private void showLive() {
        LinearLayout content = column(NAVY, true);
        add(content, text("Live Recording", 24, Color.WHITE, 1));
        add(content, text("The Android camera records a clip, then sends it to the laptop model.", 14, Color.rgb(191, 211, 228), 0));
        gap(content, 16);
        LinearLayout panel = column(Color.BLACK, true);
        panel.setGravity(Gravity.CENTER);
        add(panel, centeredText("Camera opens when you tap record", 16, Color.WHITE, 1));
        add(content, panel);
        gap(content, 14);
        LinearLayout tips = column(Color.rgb(10, 93, 146), true);
        add(tips, text("Recording Tips", 17, Color.WHITE, 1));
        add(tips, text("• Use good lighting\n• Keep your face centered\n• Hold the camera steady\n• Speak clearly and slowly", 14, Color.WHITE, 0));
        add(content, tips);
        gap(content, 20);
        Button record = button("Open Camera", BLUE, Color.WHITE);
        record.setOnClickListener(v -> openCamera());
        add(content, record);
        setScreen(content);
    }

    private void openCamera() {
        Intent intent = new Intent(MediaStore.ACTION_VIDEO_CAPTURE);
        intent.putExtra(MediaStore.EXTRA_DURATION_LIMIT, 12);
        startActivityForResult(intent, REQUEST_RECORD);
    }

    private void openUploadPicker() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.setType("video/*");
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        startActivityForResult(intent, REQUEST_UPLOAD);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK || data == null || data.getData() == null) return;
        Uri uri = data.getData();
        String source = requestCode == REQUEST_RECORD ? "live" : "upload";
        uploadVideo(uri, source);
    }

    private void uploadVideo(Uri uri, String source) {
        showProcessing("Uploading video...");
        executor.execute(() -> {
            try {
                byte[] bytes = readAll(uri);
                String boundary = "----HearMeOutAndroid" + System.currentTimeMillis();
                ByteArrayOutputStream body = new ByteArrayOutputStream();
                writePart(body, boundary, "source", source);
                writePart(body, boundary, "detector", prefs.getString("detector", "hog"));
                body.write(("--" + boundary + "\r\n").getBytes(StandardCharsets.UTF_8));
                body.write("Content-Disposition: form-data; name=\"video\"; filename=\"android-video.mp4\"\r\n".getBytes(StandardCharsets.UTF_8));
                body.write("Content-Type: video/mp4\r\n\r\n".getBytes(StandardCharsets.UTF_8));
                body.write(bytes);
                body.write("\r\n".getBytes(StandardCharsets.UTF_8));
                body.write(("--" + boundary + "--\r\n").getBytes(StandardCharsets.UTF_8));
                JSONObject job = http("POST", "/api/predict/upload", body.toByteArray(), "multipart/form-data; boundary=" + boundary);
                pollJob(job.getString("job_id"));
            } catch (Exception ex) {
                main.post(() -> {
                    toast(ex.getMessage());
                    showHome();
                });
            }
        });
    }

    private void writePart(ByteArrayOutputStream body, String boundary, String name, String value) throws Exception {
        body.write(("--" + boundary + "\r\n").getBytes(StandardCharsets.UTF_8));
        body.write(("Content-Disposition: form-data; name=\"" + name + "\"\r\n\r\n").getBytes(StandardCharsets.UTF_8));
        body.write((value + "\r\n").getBytes(StandardCharsets.UTF_8));
    }

    private byte[] readAll(Uri uri) throws Exception {
        try (InputStream input = getContentResolver().openInputStream(uri);
             ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[8192];
            int read;
            while ((read = input.read(buffer)) != -1) out.write(buffer, 0, read);
            return out.toByteArray();
        }
    }

    private void showProcessing(String msg) {
        LinearLayout root = column(BLUE, true);
        root.setGravity(Gravity.CENTER);
        add(root, centeredText("Processing Video", 26, Color.WHITE, 1));
        add(root, centeredText("Analyzing lip movements...", 15, Color.WHITE, 0));
        ProgressBar progress = new ProgressBar(this);
        root.addView(progress, new LinearLayout.LayoutParams(dp(92), dp(92)));
        add(root, centeredText(msg, 15, Color.WHITE, 1));
        setScreen(root);
    }

    private void pollJob(String jobId) {
        main.post(() -> showProcessing("AI analyzing ..."));
        executor.execute(() -> {
            try {
                JSONObject job = null;
                for (int i = 0; i < 900; i++) {
                    JSONObject response = http("GET", "/api/jobs/" + jobId, null, null);
                    job = response.getJSONObject("job");
                    String status = job.optString("status");
                    if ("succeeded".equals(status) || "failed".equals(status)) break;
                    Thread.sleep(1600);
                }
                if (job == null || !"succeeded".equals(job.optString("status"))) {
                    throw new RuntimeException(job == null ? "Prediction timed out" : job.optString("error", "Prediction failed"));
                }
                lastResult = job.getJSONObject("result");
                main.post(() -> showResult());
            } catch (Exception ex) {
                main.post(() -> {
                    toast(ex.getMessage());
                    showHome();
                });
            }
        });
    }

    private void showResult() {
        LinearLayout content = column(NAVY, true);
        add(content, text("Prediction Result", 24, Color.WHITE, 1));
        add(content, text("Lip analysis completed successfully.", 14, Color.rgb(191, 211, 228), 0));
        gap(content, 12);
        LinearLayout card = column(PANEL, true);
        add(card, text("Predicted Speech", 18, Color.rgb(16, 34, 53), 1));
        add(card, text(lastResult == null ? "" : lastResult.optString("predicted_text"), 18, Color.rgb(16, 34, 53), 0));
        gap(card, 12);
        add(card, text("Confidence Score", 17, Color.rgb(16, 34, 53), 1));
        add(card, text("Confidence is not exposed by this AV-HuBERT decoder.", 13, Color.rgb(80, 105, 128), 0));
        add(content, card);
        gap(content, 12);
        Button save = button("Save Result", BLUE, Color.WHITE);
        save.setOnClickListener(v -> toast("Already saved to history."));
        add(content, save);
        gap(content, 8);
        Button again = button("Try again", Color.LTGRAY, Color.rgb(16, 34, 53));
        again.setOnClickListener(v -> showHome());
        add(content, again);
        gap(content, 8);
        Button export = button("Export Text", Color.LTGRAY, Color.rgb(16, 34, 53));
        export.setOnClickListener(v -> shareResult());
        add(content, export);
        gap(content, 14);
        Button translate = button("Translate English to Arabic", BLUE, Color.WHITE);
        translate.setOnClickListener(v -> translateResult());
        add(content, translate);
        gap(content, 8);
        Button speak = button("Play Audio", BLUE, Color.WHITE);
        speak.setOnClickListener(v -> speak(lastResult == null ? "" : lastResult.optString("predicted_text")));
        add(content, speak);
        setScreen(content);
    }

    private void translateResult() {
        if (lastResult == null) return;
        JSONObject body = new JSONObject();
        try {
            body.put("text", lastResult.optString("predicted_text"));
            body.put("prediction_id", lastResult.optInt("id"));
        } catch (Exception ignored) {
        }
        call("POST", "/api/translate", body, null, result -> {
            String arabic = result.optString("arabic_text");
            lastResult = copyWith(lastResult, "arabic_text", arabic);
            speak(arabic);
            toast(arabic);
        }, this::toast);
    }

    private JSONObject copyWith(JSONObject original, String key, String value) {
        try {
            JSONObject copy = new JSONObject(original.toString());
            copy.put(key, value);
            return copy;
        } catch (Exception ex) {
            return original;
        }
    }

    private void speak(String value) {
        if (value == null || value.trim().isEmpty()) return;
        tts.speak(value, TextToSpeech.QUEUE_FLUSH, null, "hearmeout");
    }

    private void shareResult() {
        if (lastResult == null) return;
        Intent send = new Intent(Intent.ACTION_SEND);
        send.setType("text/plain");
        send.putExtra(Intent.EXTRA_TEXT, lastResult.optString("predicted_text"));
        startActivity(Intent.createChooser(send, "Export prediction"));
    }

    private void showHistory() {
        LinearLayout content = column(NAVY, true);
        add(content, text("Prediction History", 24, Color.WHITE, 1));
        add(content, text("View and manage your previous analyses.", 14, Color.rgb(191, 211, 228), 0));
        gap(content, 14);
        call("GET", "/api/history", null, null, result -> {
            try {
                JSONArray items = result.getJSONArray("items");
                content.removeAllViews();
                add(content, text("Prediction History", 24, Color.WHITE, 1));
                gap(content, 14);
                if (items.length() == 0) {
                    add(content, centeredText("No Predictions Yet\nYour analyzed videos and predictions will appear here.", 16, Color.WHITE, 1));
                } else {
                    for (int i = 0; i < items.length(); i++) {
                        JSONObject item = items.getJSONObject(i);
                        LinearLayout card = column(PANEL, true);
                        add(card, text(item.optString("created_at"), 13, Color.rgb(80, 105, 128), 0));
                        add(card, text(item.optString("predicted_text"), 16, Color.rgb(16, 34, 53), 1));
                        add(content, card);
                        gap(content, 10);
                    }
                }
            } catch (Exception ex) {
                toast(ex.getMessage());
            }
        }, this::toast);
        setScreen(withBottomNav(content, "history"));
    }

    private void showSettings() {
        LinearLayout content = column(NAVY, true);
        add(content, text("Settings", 24, Color.WHITE, 1));
        add(content, text("Laptop server URL", 14, Color.rgb(191, 211, 228), 1));
        EditText server = input("http://192.168.1.20:8000");
        server.setText(serverUrl());
        add(content, server);
        gap(content, 12);
        Button save = button("Save server URL", BLUE, Color.WHITE);
        save.setOnClickListener(v -> {
            saveServerUrl(server.getText().toString());
            toast("Saved");
        });
        add(content, save);
        gap(content, 12);
        Button detector = button("Use HOG detector", Color.LTGRAY, Color.rgb(16, 34, 53));
        detector.setOnClickListener(v -> {
            prefs.edit().putString("detector", "hog").apply();
            toast("Detector set to HOG");
        });
        add(content, detector);
        gap(content, 8);
        Button gpu = button("Use GPU detector", Color.LTGRAY, Color.rgb(16, 34, 53));
        gpu.setOnClickListener(v -> {
            prefs.edit().putString("detector", "fa").apply();
            toast("Detector set to Face Alignment");
        });
        add(content, gpu);
        gap(content, 20);
        Button logout = button("Log out", Color.rgb(230, 80, 80), Color.WHITE);
        logout.setOnClickListener(v -> {
            prefs.edit().remove("token").apply();
            token = "";
            user = null;
            showStart();
        });
        add(content, logout);
        setScreen(withBottomNav(content, "settings"));
    }

    private View withBottomNav(LinearLayout content, String active) {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(NAVY);
        ScrollView scroll = new ScrollView(this);
        scroll.addView(content);
        root.addView(scroll, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1
        ));
        LinearLayout nav = new LinearLayout(this);
        nav.setOrientation(LinearLayout.HORIZONTAL);
        nav.setBackgroundColor(Color.rgb(236, 248, 255));
        nav.setPadding(dp(8), dp(6), dp(8), dp(6));
        nav.addView(navButton("Home", active.equals("home"), v -> showHome()));
        nav.addView(navButton("History", active.equals("history"), v -> showHistory()));
        nav.addView(navButton("Settings", active.equals("settings"), v -> showSettings()));
        root.addView(nav, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, dp(68)
        ));
        return root;
    }

    private Button navButton(String label, boolean active, View.OnClickListener listener) {
        Button button = button(label, Color.TRANSPARENT, active ? BLUE_DARK : Color.rgb(70, 100, 127));
        button.setOnClickListener(listener);
        button.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1));
        return button;
    }

    private void call(String method, String path, JSONObject body, String contentType, JsonSuccess success, ErrorFailure failure) {
        executor.execute(() -> {
            try {
                byte[] raw = body == null ? null : body.toString().getBytes(StandardCharsets.UTF_8);
                JSONObject response = http(method, path, raw, contentType == null ? "application/json" : contentType);
                main.post(() -> {
                    try {
                        success.run(response);
                    } catch (Exception ex) {
                        failure.run(ex.getMessage());
                    }
                });
            } catch (Exception ex) {
                main.post(() -> failure.run(ex.getMessage()));
            }
        });
    }

    private JSONObject http(String method, String path, byte[] body, String contentType) throws Exception {
        URL url = new URL(serverUrl() + path);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod(method);
        conn.setConnectTimeout(10000);
        conn.setReadTimeout(1800000);
        conn.setRequestProperty("Accept", "application/json");
        if (!token.isEmpty()) conn.setRequestProperty("Authorization", "Bearer " + token);
        if (body != null) {
            conn.setDoOutput(true);
            conn.setRequestProperty("Content-Type", contentType);
            conn.setFixedLengthStreamingMode(body.length);
            try (OutputStream out = conn.getOutputStream()) {
                out.write(body);
            }
        }
        int status = conn.getResponseCode();
        InputStream input = status >= 400 ? conn.getErrorStream() : conn.getInputStream();
        String text = readString(input);
        JSONObject json = text.isEmpty() ? new JSONObject() : new JSONObject(text);
        if (status >= 400) throw new RuntimeException(json.optString("error", "Request failed"));
        return json;
    }

    private String readString(InputStream input) throws Exception {
        if (input == null) return "";
        try (InputStream in = input; ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[4096];
            int read;
            while ((read = in.read(buffer)) != -1) out.write(buffer, 0, read);
            return out.toString("UTF-8");
        }
    }

    private void toast(String msg) {
        Toast.makeText(this, msg == null ? "Something went wrong" : msg, Toast.LENGTH_LONG).show();
    }

    interface JsonSuccess {
        void run(JSONObject result) throws Exception;
    }

    interface ErrorFailure {
        void run(String message);
    }
}
