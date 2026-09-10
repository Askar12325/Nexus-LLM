package proxy

import (
	"testing"
)

func TestSanitizer_RedactsSensitiveData(t *testing.T) {
	input := "Hello, my email is alice.smith@corp.com and my card is 4111-2222-3333-4444. Call me at 555-123-4567. Secret key: sk-abc12345678901234567890."

	report := SanitizePrompt(input)
	if !report.WasSanitized {
		t.Fatalf("expected text to be sanitized")
	}

	if len(report.RedactedTypes) < 4 {
		t.Fatalf("expected at least 4 redacted types, got %d: %v", len(report.RedactedTypes), report.RedactedTypes)
	}

	expectedSubstring := "[EMAIL_REDACTED]"
	if !contains(report.CleanedText, expectedSubstring) {
		t.Fatalf("expected CleanedText to contain %s, got: %s", expectedSubstring, report.CleanedText)
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > len(substr) && stringContains(s, substr))
}

func stringContains(s, substr string) bool {
	for i := 0; i+len(substr) <= len(s); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
