import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  runApp(const HearMeOutApp());
}

enum AppScreen {
  start,
  signup,
  login,
  success,
  home,
  live,
  upload,
  processing,
  result,
  history,
  settings,
}

class HearMeOutApp extends StatelessWidget {
  const HearMeOutApp({super.key});

  @override
  Widget build(BuildContext context) {
    const navy = Color(0xFF061F36);
    const blue = Color(0xFF2D6BFF);

    return MaterialApp(
      title: 'HearMeOut',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        fontFamily: 'Roboto',
        colorScheme: ColorScheme.fromSeed(
          seedColor: blue,
          brightness: Brightness.light,
          primary: blue,
          secondary: const Color(0xFF13B8F4),
          surface: const Color(0xFFF6F9FC),
        ),
        scaffoldBackgroundColor: const Color(0xFFF6F9FC),
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFFF6F9FC),
          foregroundColor: navy,
          centerTitle: false,
          elevation: 0,
        ),
        cardTheme: CardThemeData(
          elevation: 0,
          color: Colors.white,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(8),
            side: const BorderSide(color: Color(0xFFE1E9F2)),
          ),
        ),
        filledButtonTheme: FilledButtonThemeData(
          style: FilledButton.styleFrom(
            minimumSize: const Size.fromHeight(52),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(8),
            ),
            textStyle: const TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w700,
            ),
          ),
        ),
        outlinedButtonTheme: OutlinedButtonThemeData(
          style: OutlinedButton.styleFrom(
            minimumSize: const Size.fromHeight(52),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(8),
            ),
            textStyle: const TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w700,
            ),
          ),
        ),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: Colors.white,
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(8),
            borderSide: const BorderSide(color: Color(0xFFD4E0EB)),
          ),
          contentPadding: const EdgeInsets.symmetric(
            horizontal: 14,
            vertical: 14,
          ),
        ),
      ),
      home: const HearMeOutShell(),
    );
  }
}

class HearMeOutShell extends StatefulWidget {
  const HearMeOutShell({super.key});

  @override
  State<HearMeOutShell> createState() => _HearMeOutShellState();
}

class _HearMeOutShellState extends State<HearMeOutShell> {
  static const _tokenKey = 'hmo_token';
  static const _serverKey = 'hmo_server_url';
  static const _detectorKey = 'hmo_detector';
  static const _scanDeviceKey = 'hmo_scan_device';

  final _picker = ImagePicker();
  final _tts = FlutterTts();
  final _serverController = TextEditingController();
  final _nameController = TextEditingController();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  final _settingsServerController = TextEditingController();

  AppScreen _screen = AppScreen.start;
  UserAccount? _user;
  PredictionResult? _result;
  List<PredictionResult> _history = [];
  Timer? _pollTimer;

  String _serverUrl = '';
  String _token = '';
  String _detector = 'hog';
  String _scanDevice = 'auto';
  String _message = '';
  String _jobMessage = '';
  int _jobProgress = 0;
  int _introIndex = 0;
  bool _busy = false;
  bool _booting = true;
  bool _showArabic = false;

  @override
  void initState() {
    super.initState();
    unawaited(_boot());
  }

  @override
  void dispose() {
    _pollTimer?.cancel();
    _serverController.dispose();
    _nameController.dispose();
    _emailController.dispose();
    _passwordController.dispose();
    _settingsServerController.dispose();
    unawaited(_tts.stop());
    super.dispose();
  }

  Future<void> _boot() async {
    final prefs = await SharedPreferences.getInstance();
    _serverUrl = prefs.getString(_serverKey) ?? '';
    _token = prefs.getString(_tokenKey) ?? '';
    _detector = prefs.getString(_detectorKey) ?? 'hog';
    _scanDevice = prefs.getString(_scanDeviceKey) ?? 'auto';
    _serverController.text = _serverUrl;
    _settingsServerController.text = _serverUrl;

    if (_serverUrl.isNotEmpty && _token.isNotEmpty) {
      try {
        final data = await _getJson('/api/me');
        _user = UserAccount.fromJson(data['user'] as Map<String, dynamic>);
        await _loadSettings(silent: true);
        _screen = AppScreen.home;
      } catch (_) {
        await prefs.remove(_tokenKey);
        _token = '';
        _screen = AppScreen.start;
      }
    }

    if (mounted) {
      setState(() {
        _booting = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_booting) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      appBar: _usesAppBar ? _buildAppBar() : null,
      body: SafeArea(child: _buildCurrentScreen()),
      bottomNavigationBar: _usesBottomNav ? _buildBottomNav() : null,
    );
  }

  bool get _usesAppBar {
    return {
      AppScreen.home,
      AppScreen.live,
      AppScreen.upload,
      AppScreen.result,
      AppScreen.history,
      AppScreen.settings,
    }.contains(_screen);
  }

  bool get _usesBottomNav {
    return {
      AppScreen.home,
      AppScreen.history,
      AppScreen.settings,
    }.contains(_screen);
  }

  PreferredSizeWidget _buildAppBar() {
    final title = switch (_screen) {
      AppScreen.live => 'Live Recording',
      AppScreen.upload => 'Upload Video',
      AppScreen.result => 'Prediction Result',
      AppScreen.history => 'History',
      AppScreen.settings => 'Settings',
      _ => 'HearMeOut',
    };

    return AppBar(
      title: Text(title, style: const TextStyle(fontWeight: FontWeight.w800)),
      leading:
          _screen == AppScreen.home ||
              _screen == AppScreen.history ||
              _screen == AppScreen.settings
          ? null
          : IconButton(
              icon: const Icon(Icons.arrow_back),
              tooltip: 'Back',
              onPressed: () => _goTo(AppScreen.home),
            ),
      actions: [
        if (_screen != AppScreen.settings && _token.isNotEmpty)
          IconButton(
            icon: const Icon(Icons.settings_outlined),
            tooltip: 'Settings',
            onPressed: () => _goTo(AppScreen.settings),
          ),
      ],
    );
  }

  Widget _buildCurrentScreen() {
    return switch (_screen) {
      AppScreen.start => _buildStart(),
      AppScreen.signup => _buildAuth(isSignup: true),
      AppScreen.login => _buildAuth(isSignup: false),
      AppScreen.success => _buildSuccess(),
      AppScreen.home => _buildHome(),
      AppScreen.live => _buildLive(),
      AppScreen.upload => _buildUpload(),
      AppScreen.processing => _buildProcessing(),
      AppScreen.result => _buildResult(),
      AppScreen.history => _buildHistory(),
      AppScreen.settings => _buildSettings(),
    };
  }

  Widget _buildStart() {
    final slides = [
      const IntroSlide(
        icon: Icons.auto_awesome,
        title: 'HearMeOut',
        copy: 'AI-powered lip reading for live recordings and uploaded videos.',
      ),
      const IntroSlide(
        icon: Icons.visibility_outlined,
        title: 'Read Lips with AI',
        copy:
            'The phone captures the clip, and your laptop runs the AV-HuBERT model.',
      ),
      const IntroSlide(
        icon: Icons.video_camera_back_outlined,
        title: 'Record or Upload',
        copy:
            'Use the Android camera for live capture or choose a prerecorded video.',
      ),
      const IntroSlide(
        icon: Icons.translate_outlined,
        title: 'Results and Tools',
        copy:
            'Review predictions, translate to Arabic, play text-to-speech, and save history.',
      ),
    ];
    final slide = slides[_introIndex];
    final isLast = _introIndex == slides.length - 1;

    return ListView(
      padding: const EdgeInsets.fromLTRB(22, 18, 22, 28),
      children: [
        const SizedBox(height: 20),
        const Center(child: LogoPanel()),
        const SizedBox(height: 30),
        AnimatedSwitcher(
          duration: const Duration(milliseconds: 240),
          child: Column(
            key: ValueKey(_introIndex),
            children: [
              Icon(slide.icon, size: 52, color: const Color(0xFF2D6BFF)),
              const SizedBox(height: 14),
              Text(
                slide.title,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  fontSize: 28,
                  fontWeight: FontWeight.w900,
                  color: Color(0xFF061F36),
                ),
              ),
              const SizedBox(height: 10),
              Text(
                slide.copy,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  fontSize: 16,
                  height: 1.45,
                  color: Color(0xFF486174),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 22),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: List.generate(
            slides.length,
            (index) => Container(
              width: index == _introIndex ? 22 : 8,
              height: 8,
              margin: const EdgeInsets.symmetric(horizontal: 4),
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(20),
                color: index == _introIndex
                    ? const Color(0xFF2D6BFF)
                    : const Color(0xFFC9D7E5),
              ),
            ),
          ),
        ),
        const SizedBox(height: 28),
        if (isLast) ...[
          FilledButton.icon(
            onPressed: () => _goTo(AppScreen.signup),
            icon: const Icon(Icons.person_add_alt_1),
            label: const Text('Sign Up'),
          ),
          const SizedBox(height: 12),
          OutlinedButton.icon(
            onPressed: () => _goTo(AppScreen.login),
            icon: const Icon(Icons.login),
            label: const Text('Login'),
          ),
        ] else ...[
          FilledButton(
            onPressed: () => setState(() => _introIndex += 1),
            child: Text(_introIndex == 0 ? 'Start' : 'Next'),
          ),
          TextButton(
            onPressed: () => setState(() => _introIndex = slides.length - 1),
            child: const Text('Skip'),
          ),
        ],
      ],
    );
  }

  Widget _buildAuth({required bool isSignup}) {
    final title = isSignup ? 'Sign Up' : 'Log In';
    final subtitle = isSignup
        ? 'Create an account saved in the SQLite database on your laptop.'
        : 'Use the account saved on your laptop backend.';

    return Form(
      child: ListView(
        padding: const EdgeInsets.fromLTRB(22, 22, 22, 28),
        children: [
          const Center(child: LogoMark(size: 86)),
          const SizedBox(height: 26),
          Text(
            title,
            style: const TextStyle(
              fontSize: 30,
              fontWeight: FontWeight.w900,
              color: Color(0xFF061F36),
            ),
          ),
          const SizedBox(height: 8),
          Text(
            subtitle,
            style: const TextStyle(fontSize: 15, color: Color(0xFF546D7E)),
          ),
          const SizedBox(height: 22),
          TextField(
            controller: _serverController,
            keyboardType: TextInputType.url,
            textInputAction: TextInputAction.next,
            decoration: const InputDecoration(
              labelText: 'Laptop server URL',
              hintText: 'http://192.168.1.20:8000',
              prefixIcon: Icon(Icons.dns_outlined),
            ),
          ),
          const SizedBox(height: 8),
          const Text(
            'Use the phone URL printed by run_mobile_app.ps1. On a real phone, localhost means the phone, not the laptop.',
            style: TextStyle(
              fontSize: 12,
              height: 1.35,
              color: Color(0xFF667D8E),
            ),
          ),
          const SizedBox(height: 16),
          if (isSignup) ...[
            TextField(
              controller: _nameController,
              textInputAction: TextInputAction.next,
              decoration: const InputDecoration(
                labelText: 'Name',
                prefixIcon: Icon(Icons.badge_outlined),
              ),
            ),
            const SizedBox(height: 14),
          ],
          TextField(
            controller: _emailController,
            keyboardType: TextInputType.emailAddress,
            textInputAction: TextInputAction.next,
            decoration: const InputDecoration(
              labelText: 'Email',
              hintText: 'you@example.com',
              prefixIcon: Icon(Icons.mail_outline),
            ),
          ),
          const SizedBox(height: 14),
          TextField(
            controller: _passwordController,
            obscureText: true,
            decoration: const InputDecoration(
              labelText: 'Password',
              prefixIcon: Icon(Icons.lock_outline),
            ),
            onSubmitted: (_) => isSignup ? _submitSignup() : _submitLogin(),
          ),
          if (_message.isNotEmpty) ...[
            const SizedBox(height: 14),
            StatusBanner(message: _message),
          ],
          const SizedBox(height: 22),
          FilledButton.icon(
            onPressed: _busy ? null : (isSignup ? _submitSignup : _submitLogin),
            icon: _busy
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : Icon(isSignup ? Icons.person_add_alt_1 : Icons.login),
            label: Text(isSignup ? 'Create Account' : 'Log In'),
          ),
          const SizedBox(height: 12),
          TextButton(
            onPressed: _busy
                ? null
                : () => _goTo(isSignup ? AppScreen.login : AppScreen.signup),
            child: Text(
              isSignup
                  ? 'Already have an account? Log in'
                  : 'Do not have an account? Sign up',
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSuccess() {
    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 118,
              height: 118,
              decoration: const BoxDecoration(
                color: Color(0xFFE7F9EE),
                shape: BoxShape.circle,
              ),
              child: const Icon(
                Icons.check_rounded,
                size: 70,
                color: Color(0xFF18A957),
              ),
            ),
            const SizedBox(height: 26),
            const Text(
              'Success',
              style: TextStyle(
                fontSize: 34,
                fontWeight: FontWeight.w900,
                color: Color(0xFF061F36),
              ),
            ),
            const SizedBox(height: 8),
            const Text(
              'Congratulations, you have completed your registration.',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 16, color: Color(0xFF546D7E)),
            ),
            const SizedBox(height: 28),
            FilledButton(
              onPressed: () => _goTo(AppScreen.home),
              child: const Text('Done'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHome() {
    final name = _user?.displayName.split(' ').first ?? 'there';

    return ListView(
      padding: const EdgeInsets.fromLTRB(18, 10, 18, 24),
      children: [
        Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Hi, $name',
                    style: const TextStyle(
                      fontSize: 28,
                      fontWeight: FontWeight.w900,
                      color: Color(0xFF061F36),
                    ),
                  ),
                  const SizedBox(height: 6),
                  const Text(
                    'Analyze lip movements using the model running on your laptop.',
                    style: TextStyle(color: Color(0xFF546D7E), height: 1.35),
                  ),
                ],
              ),
            ),
            CircleAvatar(
              radius: 25,
              backgroundColor: const Color(0xFF2D6BFF),
              child: Text(
                name.isEmpty ? 'H' : name[0].toUpperCase(),
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 20,
                  fontWeight: FontWeight.w900,
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 22),
        ActionTile(
          icon: Icons.fiber_manual_record,
          title: 'Record Live',
          subtitle:
              'Open the phone camera, capture a clip, and send it to the laptop.',
          color: const Color(0xFFE43F5A),
          onTap: () => _goTo(AppScreen.live),
        ),
        const SizedBox(height: 12),
        ActionTile(
          icon: Icons.upload_file_outlined,
          title: 'Upload Video',
          subtitle: 'Analyze a prerecorded video stored on your Android phone.',
          color: const Color(0xFF2D6BFF),
          onTap: () => _goTo(AppScreen.upload),
        ),
        const SizedBox(height: 12),
        ActionTile(
          icon: Icons.history,
          title: 'History',
          subtitle: 'View predictions saved in the laptop database.',
          color: const Color(0xFF00A88F),
          onTap: () => _goTo(AppScreen.history),
        ),
        const SizedBox(height: 18),
        const TipsPanel(),
      ],
    );
  }

  Widget _buildLive() {
    return ListView(
      padding: const EdgeInsets.fromLTRB(18, 10, 18, 28),
      children: [
        const CameraPanel(),
        const SizedBox(height: 16),
        const TipsPanel(
          title: 'Recording Tips',
          tips: [
            'Use good lighting.',
            'Keep your face centered.',
            'Hold the phone steady.',
            'Speak clearly and slowly.',
          ],
        ),
        if (_message.isNotEmpty) ...[
          const SizedBox(height: 14),
          StatusBanner(message: _message),
        ],
        const SizedBox(height: 18),
        FilledButton.icon(
          onPressed: _busy ? null : _captureLiveVideo,
          icon: const Icon(Icons.videocam_outlined),
          label: const Text('Open Camera'),
        ),
      ],
    );
  }

  Widget _buildUpload() {
    return ListView(
      padding: const EdgeInsets.fromLTRB(18, 10, 18, 28),
      children: [
        Card(
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              children: [
                Container(
                  width: 82,
                  height: 82,
                  decoration: BoxDecoration(
                    color: const Color(0xFFEAF2FF),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: const Icon(
                    Icons.upload_file_outlined,
                    size: 44,
                    color: Color(0xFF2D6BFF),
                  ),
                ),
                const SizedBox(height: 16),
                const Text(
                  'Choose a clear video where the speaker faces the camera.',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 16,
                    height: 1.4,
                    color: Color(0xFF435A6D),
                  ),
                ),
                const SizedBox(height: 18),
                FilledButton.icon(
                  onPressed: _busy ? null : _pickVideo,
                  icon: const Icon(Icons.folder_open_outlined),
                  label: const Text('Choose Video'),
                ),
              ],
            ),
          ),
        ),
        if (_message.isNotEmpty) ...[
          const SizedBox(height: 14),
          StatusBanner(message: _message),
        ],
        const SizedBox(height: 16),
        const TipsPanel(
          tips: [
            'Use short clips when possible.',
            'Keep the mouth visible.',
            'Avoid strong shadows.',
            'Make sure the face is not too small in frame.',
          ],
        ),
      ],
    );
  }

  Widget _buildProcessing() {
    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(22),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text(
              'Processing Video',
              style: TextStyle(
                fontSize: 28,
                fontWeight: FontWeight.w900,
                color: Color(0xFF061F36),
              ),
            ),
            const SizedBox(height: 8),
            const Text(
              'The laptop is analyzing lip movements.',
              textAlign: TextAlign.center,
              style: TextStyle(color: Color(0xFF546D7E)),
            ),
            const SizedBox(height: 28),
            SizedBox(
              width: 96,
              height: 96,
              child: CircularProgressIndicator(
                strokeWidth: 8,
                value: _jobProgress > 0 ? _jobProgress / 100 : null,
              ),
            ),
            const SizedBox(height: 20),
            Text(
              _jobProgress > 0 ? '$_jobProgress% completed' : 'Queued',
              style: const TextStyle(fontWeight: FontWeight.w800),
            ),
            const SizedBox(height: 10),
            Text(
              _jobMessage.isEmpty ? 'Preparing video...' : _jobMessage,
              textAlign: TextAlign.center,
              style: const TextStyle(color: Color(0xFF546D7E)),
            ),
            const SizedBox(height: 24),
            const ProcessingSteps(),
          ],
        ),
      ),
    );
  }

  Widget _buildResult() {
    final result = _result;
    if (result == null) {
      return const Center(child: Text('No result loaded.'));
    }
    final shownText = _showArabic && result.arabicText.trim().isNotEmpty
        ? result.arabicText
        : result.predictedText;
    final confidence = result.confidence;

    return ListView(
      padding: const EdgeInsets.fromLTRB(18, 10, 18, 28),
      children: [
        Card(
          child: Padding(
            padding: const EdgeInsets.all(18),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    const Icon(Icons.movie_outlined, color: Color(0xFF2D6BFF)),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        result.inputName.isEmpty
                            ? 'Recorded Video'
                            : result.inputName,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(fontWeight: FontWeight.w800),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 18),
                const Text(
                  'Predicted Speech',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.w900),
                ),
                const SizedBox(height: 10),
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(14),
                  decoration: BoxDecoration(
                    color: const Color(0xFFF3F7FB),
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(color: const Color(0xFFD9E5EF)),
                  ),
                  child: Text(
                    shownText.isEmpty ? 'No text returned.' : shownText,
                    textDirection: _showArabic
                        ? TextDirection.rtl
                        : TextDirection.ltr,
                    style: const TextStyle(fontSize: 17, height: 1.45),
                  ),
                ),
                const SizedBox(height: 18),
                const Text(
                  'Confidence Score',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.w900),
                ),
                const SizedBox(height: 8),
                if (confidence == null)
                  const Text(
                    'Confidence is not exposed by the current AV-HuBERT decoder.',
                    style: TextStyle(color: Color(0xFF667D8E)),
                  )
                else ...[
                  LinearProgressIndicator(value: confidence.clamp(0, 1)),
                  const SizedBox(height: 8),
                  Text('${(confidence * 100).round()}%'),
                ],
              ],
            ),
          ),
        ),
        const SizedBox(height: 14),
        Row(
          children: [
            Expanded(
              child: FilledButton.icon(
                onPressed: _saveResult,
                icon: const Icon(Icons.bookmark_add_outlined),
                label: const Text('Save'),
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: OutlinedButton.icon(
                onPressed: () => _goTo(AppScreen.upload),
                icon: const Icon(Icons.refresh),
                label: const Text('Try Again'),
              ),
            ),
          ],
        ),
        const SizedBox(height: 10),
        OutlinedButton.icon(
          onPressed: _exportResult,
          icon: const Icon(Icons.ios_share),
          label: const Text('Export Text'),
        ),
        const SizedBox(height: 14),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(18),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Translation',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.w900),
                ),
                const SizedBox(height: 8),
                const Text(
                  'English to Arabic translation runs on the laptop backend.',
                  style: TextStyle(color: Color(0xFF667D8E)),
                ),
                const SizedBox(height: 14),
                SegmentedButton<bool>(
                  segments: const [
                    ButtonSegment(value: false, label: Text('English')),
                    ButtonSegment(value: true, label: Text('Arabic')),
                  ],
                  selected: {_showArabic},
                  onSelectionChanged: (values) {
                    final wantsArabic = values.first;
                    if (wantsArabic) {
                      unawaited(_translateArabic());
                    } else {
                      setState(() => _showArabic = false);
                    }
                  },
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 14),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(18),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Text to Speech',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.w900),
                ),
                const SizedBox(height: 8),
                const Text(
                  'Speech playback uses Android text-to-speech on the phone.',
                  style: TextStyle(color: Color(0xFF667D8E)),
                ),
                const SizedBox(height: 14),
                FilledButton.icon(
                  onPressed: _speakResult,
                  icon: const Icon(Icons.volume_up_outlined),
                  label: const Text('Play Audio'),
                ),
              ],
            ),
          ),
        ),
        if (_message.isNotEmpty) ...[
          const SizedBox(height: 14),
          StatusBanner(message: _message),
        ],
      ],
    );
  }

  Widget _buildHistory() {
    if (_busy && _history.isEmpty) {
      return const Center(child: CircularProgressIndicator());
    }

    if (_history.isEmpty) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(30),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 92,
                height: 92,
                decoration: BoxDecoration(
                  color: const Color(0xFFEAF2FF),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: const Icon(
                  Icons.history,
                  size: 46,
                  color: Color(0xFF2D6BFF),
                ),
              ),
              const SizedBox(height: 18),
              const Text(
                'No Predictions Yet',
                style: TextStyle(fontSize: 24, fontWeight: FontWeight.w900),
              ),
              const SizedBox(height: 8),
              const Text(
                'Your analyzed videos and predictions will appear here.',
                textAlign: TextAlign.center,
                style: TextStyle(color: Color(0xFF667D8E)),
              ),
            ],
          ),
        ),
      );
    }

    return RefreshIndicator(
      onRefresh: _loadHistory,
      child: ListView.separated(
        padding: const EdgeInsets.fromLTRB(18, 10, 18, 28),
        itemCount: _history.length,
        separatorBuilder: (context, index) => const SizedBox(height: 10),
        itemBuilder: (context, index) {
          final item = _history[index];
          return Card(
            child: InkWell(
              borderRadius: BorderRadius.circular(8),
              onTap: () {
                setState(() {
                  _result = item;
                  _showArabic = item.arabicText.isNotEmpty;
                  _message = '';
                  _screen = AppScreen.result;
                });
              },
              child: Padding(
                padding: const EdgeInsets.all(14),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Expanded(
                          child: Text(
                            item.createdAt.isEmpty
                                ? item.inputName
                                : item.createdAt,
                            style: const TextStyle(
                              fontWeight: FontWeight.w800,
                              color: Color(0xFF061F36),
                            ),
                          ),
                        ),
                        IconButton(
                          icon: const Icon(Icons.delete_outline),
                          tooltip: 'Delete',
                          onPressed: () => _deleteHistoryItem(item.id),
                        ),
                      ],
                    ),
                    Text(
                      item.predictedText,
                      maxLines: 3,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(height: 1.35),
                    ),
                    if (item.arabicText.isNotEmpty) ...[
                      const SizedBox(height: 8),
                      Text(
                        item.arabicText,
                        textDirection: TextDirection.rtl,
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ],
                  ],
                ),
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildSettings() {
    return ListView(
      padding: const EdgeInsets.fromLTRB(18, 10, 18, 28),
      children: [
        Card(
          child: Padding(
            padding: const EdgeInsets.all(18),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Laptop Connection',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.w900),
                ),
                const SizedBox(height: 12),
                TextField(
                  controller: _settingsServerController,
                  keyboardType: TextInputType.url,
                  decoration: const InputDecoration(
                    labelText: 'Laptop server URL',
                    hintText: 'http://192.168.1.20:8000',
                    prefixIcon: Icon(Icons.dns_outlined),
                  ),
                ),
                const SizedBox(height: 10),
                const Text(
                  'Start run_mobile_app.ps1 on the laptop, then enter the phone URL printed by PowerShell.',
                  style: TextStyle(
                    fontSize: 12,
                    height: 1.35,
                    color: Color(0xFF667D8E),
                  ),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 12),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(18),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Model Settings',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.w900),
                ),
                const SizedBox(height: 14),
                DropdownButtonFormField<String>(
                  initialValue: _detector,
                  decoration: const InputDecoration(
                    labelText: 'Mouth detector',
                    prefixIcon: Icon(Icons.face_retouching_natural_outlined),
                  ),
                  items: const [
                    DropdownMenuItem(
                      value: 'hog',
                      child: Text('HOG CPU detector'),
                    ),
                    DropdownMenuItem(
                      value: 'fa',
                      child: Text('Face Alignment detector'),
                    ),
                  ],
                  onChanged: (value) {
                    if (value != null) setState(() => _detector = value);
                  },
                ),
                const SizedBox(height: 14),
                DropdownButtonFormField<String>(
                  initialValue: _scanDevice,
                  decoration: const InputDecoration(
                    labelText: 'GPU detector device',
                    prefixIcon: Icon(Icons.memory_outlined),
                  ),
                  items: const [
                    DropdownMenuItem(value: 'auto', child: Text('Auto')),
                    DropdownMenuItem(value: 'cpu', child: Text('CPU')),
                    DropdownMenuItem(value: 'cuda', child: Text('CUDA')),
                  ],
                  onChanged: (value) {
                    if (value != null) setState(() => _scanDevice = value);
                  },
                ),
                const SizedBox(height: 18),
                FilledButton.icon(
                  onPressed: _saveSettings,
                  icon: const Icon(Icons.save_outlined),
                  label: const Text('Save Settings'),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 12),
        OutlinedButton.icon(
          onPressed: _logout,
          icon: const Icon(Icons.logout),
          label: const Text('Log Out'),
        ),
        if (_message.isNotEmpty) ...[
          const SizedBox(height: 14),
          StatusBanner(message: _message),
        ],
      ],
    );
  }

  NavigationBar _buildBottomNav() {
    final index = switch (_screen) {
      AppScreen.history => 1,
      AppScreen.settings => 2,
      _ => 0,
    };

    return NavigationBar(
      selectedIndex: index,
      onDestinationSelected: (selected) {
        if (selected == 0) _goTo(AppScreen.home);
        if (selected == 1) _goTo(AppScreen.history);
        if (selected == 2) _goTo(AppScreen.settings);
      },
      destinations: const [
        NavigationDestination(
          icon: Icon(Icons.home_outlined),
          selectedIcon: Icon(Icons.home),
          label: 'Home',
        ),
        NavigationDestination(
          icon: Icon(Icons.history_outlined),
          selectedIcon: Icon(Icons.history),
          label: 'History',
        ),
        NavigationDestination(
          icon: Icon(Icons.settings_outlined),
          selectedIcon: Icon(Icons.settings),
          label: 'Settings',
        ),
      ],
    );
  }

  Future<void> _submitSignup() async {
    await _auth(
      path: '/api/signup',
      payload: {
        'display_name': _nameController.text.trim(),
        'email': _emailController.text.trim(),
        'password': _passwordController.text,
      },
    );
  }

  Future<void> _submitLogin() async {
    await _auth(
      path: '/api/login',
      payload: {
        'email': _emailController.text.trim(),
        'password': _passwordController.text,
      },
    );
  }

  Future<void> _auth({
    required String path,
    required Map<String, dynamic> payload,
  }) async {
    final url = _normalizeServerUrl(_serverController.text);
    if (url.isEmpty) {
      setState(() => _message = 'Enter the laptop server URL first.');
      return;
    }

    setState(() {
      _busy = true;
      _message = '';
    });

    try {
      await _setServerUrl(url);
      final data = await _postJson(path, payload);
      await _acceptAuth(data);
    } catch (error) {
      if (mounted) {
        setState(() => _message = _friendlyError(error));
      }
    } finally {
      if (mounted) {
        setState(() => _busy = false);
      }
    }
  }

  Future<void> _acceptAuth(Map<String, dynamic> data) async {
    final prefs = await SharedPreferences.getInstance();
    _token = data['token'] as String? ?? '';
    _user = UserAccount.fromJson(data['user'] as Map<String, dynamic>);
    await prefs.setString(_tokenKey, _token);
    if (mounted) {
      setState(() {
        _message = '';
        _screen = AppScreen.success;
      });
    }
  }

  Future<void> _captureLiveVideo() async {
    try {
      final video = await _picker.pickVideo(
        source: ImageSource.camera,
        preferredCameraDevice: CameraDevice.front,
        maxDuration: const Duration(seconds: 45),
      );
      if (video == null) {
        setState(() => _message = 'Recording cancelled.');
        return;
      }
      await _startPrediction(video, source: 'live');
    } catch (error) {
      setState(() => _message = _friendlyError(error));
    }
  }

  Future<void> _pickVideo() async {
    try {
      final video = await _picker.pickVideo(source: ImageSource.gallery);
      if (video == null) {
        setState(() => _message = 'No video selected.');
        return;
      }
      await _startPrediction(video, source: 'upload');
    } catch (error) {
      setState(() => _message = _friendlyError(error));
    }
  }

  Future<void> _startPrediction(XFile file, {required String source}) async {
    _pollTimer?.cancel();
    setState(() {
      _busy = true;
      _screen = AppScreen.processing;
      _jobProgress = 0;
      _jobMessage = 'Uploading video...';
      _message = '';
      _result = null;
      _showArabic = false;
    });

    try {
      final request = http.MultipartRequest(
        'POST',
        _uri('/api/predict/upload'),
      );
      request.headers.addAll(_authHeaders());
      request.fields['source'] = source;
      request.fields['detector'] = _detector;

      final length = await file.length();
      final filename = file.name.isNotEmpty ? file.name : _basename(file.path);
      request.files.add(
        http.MultipartFile(
          'video',
          http.ByteStream(file.openRead()),
          length,
          filename: filename.isEmpty ? 'video.mp4' : filename,
        ),
      );

      final streamed = await request.send();
      final response = await http.Response.fromStream(streamed);
      final data = _decodeResponse(response);
      final jobId = data['job_id'] as String?;
      if (jobId == null || jobId.isEmpty) {
        throw HearMeOutException(
          'The laptop did not return a prediction job id.',
        );
      }
      await _pollJob(jobId);
    } catch (error) {
      if (mounted) {
        setState(() {
          _busy = false;
          _message = _friendlyError(error);
          _screen = source == 'live' ? AppScreen.live : AppScreen.upload;
        });
      }
    }
  }

  Future<void> _pollJob(String jobId) async {
    try {
      final data = await _getJson('/api/jobs/$jobId');
      final job = data['job'] as Map<String, dynamic>;
      final status = job['status'] as String? ?? 'queued';
      final progress = (job['progress'] as num?)?.round() ?? 0;
      final message = job['message'] as String? ?? '';

      if (!mounted) return;
      setState(() {
        _jobProgress = progress.clamp(0, 100);
        _jobMessage = message;
      });

      if (status == 'succeeded') {
        final result = PredictionResult.fromJson(
          job['result'] as Map<String, dynamic>,
        );
        setState(() {
          _busy = false;
          _result = result;
          _screen = AppScreen.result;
        });
        return;
      }

      if (status == 'failed') {
        throw HearMeOutException(
          job['error'] as String? ??
              job['message'] as String? ??
              'Prediction failed.',
        );
      }

      _pollTimer = Timer(const Duration(milliseconds: 1600), () {
        unawaited(_pollJob(jobId));
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _busy = false;
        _message = _friendlyError(error);
        _screen = AppScreen.home;
      });
    }
  }

  Future<void> _translateArabic() async {
    final result = _result;
    if (result == null) return;
    if (result.arabicText.trim().isNotEmpty) {
      setState(() => _showArabic = true);
      return;
    }

    setState(() => _message = 'Loading Arabic translation...');
    try {
      final data = await _postJson('/api/translate', {
        'text': result.predictedText,
        'prediction_id': result.id,
      });
      final translated = data['arabic_text'] as String? ?? '';
      if (!mounted) return;
      setState(() {
        _result = result.copyWith(arabicText: translated);
        _showArabic = true;
        _message = '';
      });
    } catch (error) {
      if (mounted) {
        setState(() => _message = _friendlyError(error));
      }
    }
  }

  Future<void> _speakResult() async {
    final result = _result;
    if (result == null) return;
    final text = _showArabic && result.arabicText.isNotEmpty
        ? result.arabicText
        : result.predictedText;
    if (text.trim().isEmpty) {
      setState(() => _message = 'There is no text to read.');
      return;
    }

    await _tts.stop();
    await _tts.setLanguage(_showArabic ? 'ar' : 'en-US');
    await _tts.setSpeechRate(0.45);
    await _tts.speak(text);
  }

  void _saveResult() {
    setState(() => _message = 'Result is already saved to laptop history.');
  }

  Future<void> _exportResult() async {
    final result = _result;
    if (result == null) return;
    final text = [
      'HearMeOut Prediction',
      '',
      result.predictedText,
      if (result.arabicText.isNotEmpty) '',
      if (result.arabicText.isNotEmpty) 'Arabic: ${result.arabicText}',
    ].join('\n');
    await Clipboard.setData(ClipboardData(text: text));
    if (mounted) {
      setState(() => _message = 'Prediction text copied to clipboard.');
    }
  }

  Future<void> _loadHistory() async {
    setState(() {
      _busy = true;
      _message = '';
    });
    try {
      final data = await _getJson('/api/history');
      final items = (data['items'] as List<dynamic>? ?? [])
          .whereType<Map<String, dynamic>>()
          .map(PredictionResult.fromJson)
          .toList();
      if (mounted) {
        setState(() => _history = items);
      }
    } catch (error) {
      if (mounted) setState(() => _message = _friendlyError(error));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _deleteHistoryItem(int id) async {
    try {
      await _delete('/api/history/$id');
      if (mounted) {
        setState(() => _history.removeWhere((item) => item.id == id));
      }
    } catch (error) {
      if (mounted) setState(() => _message = _friendlyError(error));
    }
  }

  Future<void> _loadSettings({bool silent = false}) async {
    try {
      final data = await _getJson('/api/settings');
      final settings = data['settings'] as Map<String, dynamic>? ?? {};
      _detector = settings['detector'] as String? ?? _detector;
      _scanDevice = settings['scan_device'] as String? ?? _scanDevice;
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(_detectorKey, _detector);
      await prefs.setString(_scanDeviceKey, _scanDevice);
      if (mounted && !silent) setState(() {});
    } catch (_) {
      if (mounted && !silent) {
        setState(() => _message = 'Could not load laptop settings.');
      }
    }
  }

  Future<void> _saveSettings() async {
    final url = _normalizeServerUrl(_settingsServerController.text);
    if (url.isEmpty) {
      setState(() => _message = 'Enter the laptop server URL first.');
      return;
    }

    setState(() {
      _busy = true;
      _message = '';
    });

    try {
      await _setServerUrl(url);
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(_detectorKey, _detector);
      await prefs.setString(_scanDeviceKey, _scanDevice);
      if (_token.isNotEmpty) {
        await _postJson('/api/settings', {
          'detector': _detector,
          'scan_device': _scanDevice,
        });
      }
      if (mounted) {
        setState(() => _message = 'Settings saved.');
      }
    } catch (error) {
      if (mounted) {
        setState(() => _message = _friendlyError(error));
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _logout() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_tokenKey);
    await _tts.stop();
    _pollTimer?.cancel();
    setState(() {
      _token = '';
      _user = null;
      _history = [];
      _result = null;
      _message = '';
      _screen = AppScreen.start;
    });
  }

  void _goTo(AppScreen screen) {
    _pollTimer?.cancel();
    setState(() {
      _screen = screen;
      _message = '';
      if (screen != AppScreen.result) _showArabic = false;
      if (screen == AppScreen.signup || screen == AppScreen.login) {
        _serverController.text = _serverUrl;
      }
      if (screen == AppScreen.settings) {
        _settingsServerController.text = _serverUrl;
      }
    });
    if (screen == AppScreen.history) unawaited(_loadHistory());
    if (screen == AppScreen.settings && _token.isNotEmpty) {
      unawaited(_loadSettings());
    }
  }

  Future<void> _setServerUrl(String url) async {
    final normalized = _normalizeServerUrl(url);
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_serverKey, normalized);
    _serverUrl = normalized;
    _serverController.text = normalized;
    _settingsServerController.text = normalized;
  }

  Future<Map<String, dynamic>> _getJson(String path) async {
    final response = await http.get(_uri(path), headers: _authHeaders());
    return _decodeResponse(response);
  }

  Future<Map<String, dynamic>> _postJson(
    String path,
    Map<String, dynamic> body,
  ) async {
    final response = await http.post(
      _uri(path),
      headers: {..._authHeaders(), 'Content-Type': 'application/json'},
      body: jsonEncode(body),
    );
    return _decodeResponse(response);
  }

  Future<void> _delete(String path) async {
    final response = await http.delete(_uri(path), headers: _authHeaders());
    _decodeResponse(response);
  }

  Uri _uri(String path) {
    if (_serverUrl.isEmpty) {
      throw HearMeOutException('Set the laptop server URL first.');
    }
    return Uri.parse('$_serverUrl$path');
  }

  Map<String, String> _authHeaders() {
    if (_token.isEmpty) return {};
    return {'Authorization': 'Bearer $_token'};
  }

  Map<String, dynamic> _decodeResponse(http.Response response) {
    final decoded = response.body.isEmpty
        ? <String, dynamic>{}
        : jsonDecode(utf8.decode(response.bodyBytes)) as Map<String, dynamic>;
    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw HearMeOutException(
        decoded['error'] as String? ??
            'Request failed (${response.statusCode}).',
      );
    }
    return decoded;
  }

  String _normalizeServerUrl(String value) {
    var url = value.trim();
    if (url.isEmpty) return '';
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
      url = 'http://$url';
    }
    return url.replaceAll(RegExp(r'/+$'), '');
  }

  String _friendlyError(Object error) {
    if (error is HearMeOutException) return error.message;
    final text = error.toString().replaceFirst('Exception: ', '');
    if (text.contains('SocketException') ||
        text.contains('Connection refused')) {
      return 'Cannot reach the laptop server. Start run_mobile_app.ps1 and use the laptop Wi-Fi IP.';
    }
    return text;
  }

  String _basename(String path) {
    final clean = path.replaceAll('\\', '/');
    final index = clean.lastIndexOf('/');
    return index == -1 ? clean : clean.substring(index + 1);
  }
}

class HearMeOutException implements Exception {
  const HearMeOutException(this.message);
  final String message;

  @override
  String toString() => message;
}

class IntroSlide {
  const IntroSlide({
    required this.icon,
    required this.title,
    required this.copy,
  });

  final IconData icon;
  final String title;
  final String copy;
}

class UserAccount {
  const UserAccount({
    required this.id,
    required this.email,
    required this.displayName,
  });

  factory UserAccount.fromJson(Map<String, dynamic> json) {
    return UserAccount(
      id: (json['id'] as num?)?.toInt() ?? 0,
      email: json['email'] as String? ?? '',
      displayName: json['display_name'] as String? ?? 'there',
    );
  }

  final int id;
  final String email;
  final String displayName;
}

class PredictionResult {
  const PredictionResult({
    required this.id,
    required this.source,
    required this.inputName,
    required this.predictedText,
    required this.arabicText,
    required this.confidence,
    required this.outputDir,
    required this.createdAt,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    return PredictionResult(
      id: (json['id'] as num?)?.toInt() ?? 0,
      source: json['source'] as String? ?? '',
      inputName: json['input_name'] as String? ?? '',
      predictedText: json['predicted_text'] as String? ?? '',
      arabicText: json['arabic_text'] as String? ?? '',
      confidence: (json['confidence'] as num?)?.toDouble(),
      outputDir: json['output_dir'] as String? ?? '',
      createdAt: json['created_at'] as String? ?? '',
    );
  }

  final int id;
  final String source;
  final String inputName;
  final String predictedText;
  final String arabicText;
  final double? confidence;
  final String outputDir;
  final String createdAt;

  PredictionResult copyWith({String? arabicText}) {
    return PredictionResult(
      id: id,
      source: source,
      inputName: inputName,
      predictedText: predictedText,
      arabicText: arabicText ?? this.arabicText,
      confidence: confidence,
      outputDir: outputDir,
      createdAt: createdAt,
    );
  }
}

class ActionTile extends StatelessWidget {
  const ActionTile({
    super.key,
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.color,
    required this.onTap,
  });

  final IconData icon;
  final String title;
  final String subtitle;
  final Color color;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: InkWell(
        borderRadius: BorderRadius.circular(8),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Row(
            children: [
              Container(
                width: 54,
                height: 54,
                decoration: BoxDecoration(
                  color: color.withAlpha(28),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Icon(icon, color: color, size: 30),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      style: const TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.w900,
                        color: Color(0xFF061F36),
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      subtitle,
                      style: const TextStyle(
                        height: 1.35,
                        color: Color(0xFF546D7E),
                      ),
                    ),
                  ],
                ),
              ),
              const Icon(Icons.chevron_right, color: Color(0xFF91A5B5)),
            ],
          ),
        ),
      ),
    );
  }
}

class TipsPanel extends StatelessWidget {
  const TipsPanel({
    super.key,
    this.title = 'Tips for Better Accuracy',
    this.tips = const [
      'Use good lighting.',
      'Keep your face centered.',
      'Avoid excessive head movement.',
      'Speak clearly and slowly.',
    ],
  });

  final String title;
  final List<String> tips;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(
                  Icons.tips_and_updates_outlined,
                  color: Color(0xFFFFB020),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    title,
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w900,
                      color: Color(0xFF061F36),
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            ...tips.map(
              (tip) => Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Icon(
                      Icons.check_circle,
                      size: 18,
                      color: Color(0xFF18A957),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(tip, style: const TextStyle(height: 1.35)),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class CameraPanel extends StatelessWidget {
  const CameraPanel({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 360,
      decoration: BoxDecoration(
        color: const Color(0xFF061F36),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Stack(
        children: [
          const Positioned.fill(
            child: Center(
              child: Icon(
                Icons.face_retouching_natural_outlined,
                size: 98,
                color: Color(0xFF6EDCFF),
              ),
            ),
          ),
          Positioned(
            top: 14,
            left: 14,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
              decoration: BoxDecoration(
                color: Colors.white.withAlpha(28),
                borderRadius: BorderRadius.circular(8),
              ),
              child: const Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.circle, size: 10, color: Color(0xFFE43F5A)),
                  SizedBox(width: 6),
                  Text(
                    'Ready to record',
                    style: TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ],
              ),
            ),
          ),
          const Positioned(
            left: 20,
            right: 20,
            bottom: 22,
            child: Text(
              'The camera opens in Android, then the clip uploads to the laptop.',
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.white70, height: 1.35),
            ),
          ),
        ],
      ),
    );
  }
}

class ProcessingSteps extends StatelessWidget {
  const ProcessingSteps({super.key});

  @override
  Widget build(BuildContext context) {
    const steps = [
      'Extracting frames',
      'Detecting mouth movement',
      'Processing speech patterns',
      'Saving prediction',
    ];

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: steps
              .map(
                (step) => Padding(
                  padding: const EdgeInsets.symmetric(vertical: 6),
                  child: Row(
                    children: [
                      const Icon(
                        Icons.check_circle,
                        size: 18,
                        color: Color(0xFF18A957),
                      ),
                      const SizedBox(width: 8),
                      Expanded(child: Text(step)),
                    ],
                  ),
                ),
              )
              .toList(),
        ),
      ),
    );
  }
}

class StatusBanner extends StatelessWidget {
  const StatusBanner({super.key, required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFFFFF8E7),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFFFFD37A)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.info_outline, color: Color(0xFF936300)),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              message,
              style: const TextStyle(height: 1.35, color: Color(0xFF614200)),
            ),
          ),
        ],
      ),
    );
  }
}

class LogoPanel extends StatelessWidget {
  const LogoPanel({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 250,
      padding: const EdgeInsets.fromLTRB(18, 22, 18, 18),
      decoration: BoxDecoration(
        color: const Color(0xFF061F36),
        borderRadius: BorderRadius.circular(8),
      ),
      child: const Column(
        mainAxisSize: MainAxisSize.min,
        children: [LogoMark(size: 138), SizedBox(height: 12), LogoWordmark()],
      ),
    );
  }
}

class LogoWordmark extends StatelessWidget {
  const LogoWordmark({super.key});

  @override
  Widget build(BuildContext context) {
    return RichText(
      text: const TextSpan(
        style: TextStyle(
          fontSize: 29,
          fontWeight: FontWeight.w900,
          color: Colors.white,
          letterSpacing: 0,
        ),
        children: [
          TextSpan(text: 'Hear'),
          TextSpan(
            text: 'Me',
            style: TextStyle(color: Color(0xFF5D74FF)),
          ),
          TextSpan(
            text: 'Out',
            style: TextStyle(color: Color(0xFF13B8F4)),
          ),
        ],
      ),
    );
  }
}

class LogoMark extends StatelessWidget {
  const LogoMark({super.key, this.size = 120});

  final double size;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: size,
      height: size * 0.72,
      child: CustomPaint(painter: LogoPainter()),
    );
  }
}

class LogoPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final sx = size.width / 140;
    final sy = size.height / 100;
    canvas.scale(sx, sy);

    final topLip = Path()
      ..moveTo(34, 49)
      ..cubicTo(50, 20, 64, 26, 73, 34)
      ..cubicTo(83, 25, 100, 19, 119, 49)
      ..cubicTo(98, 50, 90, 40, 76, 43)
      ..cubicTo(59, 47, 51, 49, 34, 49)
      ..close();
    final bottomLip = Path()
      ..moveTo(34, 60)
      ..cubicTo(56, 65, 67, 78, 79, 78)
      ..cubicTo(94, 78, 106, 65, 120, 60)
      ..cubicTo(102, 88, 87, 95, 76, 95)
      ..cubicTo(60, 95, 47, 85, 34, 60)
      ..close();

    final lipPaint = Paint()
      ..shader = const LinearGradient(
        colors: [Color(0xFF5D74FF), Color(0xFF13B8F4)],
      ).createShader(const Rect.fromLTWH(30, 20, 95, 75));
    canvas.drawPath(topLip, lipPaint);
    canvas.drawPath(bottomLip, lipPaint);

    final wave = Path()
      ..moveTo(28, 55)
      ..cubicTo(35, 55, 35, 44, 42, 44)
      ..cubicTo(49, 44, 49, 67, 56, 67)
      ..cubicTo(63, 67, 63, 40, 70, 40)
      ..cubicTo(77, 40, 77, 66, 84, 66)
      ..cubicTo(91, 66, 91, 46, 98, 46)
      ..cubicTo(105, 46, 105, 58, 114, 58);
    canvas.drawPath(
      wave,
      Paint()
        ..color = Colors.white
        ..strokeWidth = 4
        ..style = PaintingStyle.stroke
        ..strokeCap = StrokeCap.round,
    );

    final soundPaint = Paint()
      ..color = const Color(0xFF5D74FF)
      ..strokeWidth = 4
      ..strokeCap = StrokeCap.round;
    canvas.drawLine(const Offset(20, 44), const Offset(20, 66), soundPaint);
    canvas.drawLine(const Offset(14, 48), const Offset(14, 62), soundPaint);
    canvas.drawLine(const Offset(8, 52), const Offset(8, 58), soundPaint);

    final bubble = RRect.fromRectAndRadius(
      const Rect.fromLTWH(113, 35, 39, 34),
      const Radius.circular(10),
    );
    canvas.drawRRect(
      bubble,
      Paint()
        ..color = Colors.transparent
        ..style = PaintingStyle.stroke
        ..strokeWidth = 3
        ..shader = const LinearGradient(
          colors: [Color(0xFF5D74FF), Color(0xFF13B8F4)],
        ).createShader(const Rect.fromLTWH(113, 35, 39, 34)),
    );
    final linePaint = Paint()
      ..color = const Color(0xFF13B8F4)
      ..strokeWidth = 3
      ..strokeCap = StrokeCap.round;
    canvas.drawLine(const Offset(123, 46), const Offset(143, 46), linePaint);
    canvas.drawLine(const Offset(123, 55), const Offset(137, 55), linePaint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
