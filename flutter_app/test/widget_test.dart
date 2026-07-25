import 'package:flutter_test/flutter_test.dart';
import 'package:hear_me_out/main.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  testWidgets('HearMeOut starts on the onboarding screen', (tester) async {
    SharedPreferences.setMockInitialValues({});

    await tester.pumpWidget(const HearMeOutApp());
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 100));

    expect(find.text('HearMeOut'), findsWidgets);
    expect(
      find.text(
        'AI-powered lip reading for live recordings and uploaded videos.',
      ),
      findsOneWidget,
    );
  });
}
