package proxy

import (
	"regexp"
)

var (
	emailRegex      = regexp.MustCompile(`(?i)[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}`)
	creditCardRegex = regexp.MustCompile(`\b(?:\d{4}[-\s]?){3}\d{4}\b`)
	phoneRegex      = regexp.MustCompile(`\b(?:\+?\d{1,3}[-\s.]?)?\(?\d{3}\)?[-\s.]?\d{3}[-\s.]?\d{4}\b`)
	apiKeyRegex     = regexp.MustCompile(`\b(?:sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,}|nx-key-[a-zA-Z0-9]{10,})\b`)
	ssnRegex        = regexp.MustCompile(`\b\d{3}-\d{2}-\d{4}\b`)
)

// SanitizeReport details what PII was detected and redacted
type SanitizeReport struct {
	CleanedText  string   `json:"cleaned_text"`
	RedactedTypes []string `json:"redacted_types"`
	WasSanitized bool     `json:"was_sanitized"`
}

// SanitizePrompt scrubs sensitive PII patterns from text before forwarding upstream
func SanitizePrompt(text string) SanitizeReport {
	original := text
	var types []string

	if emailRegex.MatchString(text) {
		text = emailRegex.ReplaceAllString(text, "[EMAIL_REDACTED]")
		types = append(types, "EMAIL")
	}
	if creditCardRegex.MatchString(text) {
		text = creditCardRegex.ReplaceAllString(text, "[CREDIT_CARD_REDACTED]")
		types = append(types, "CREDIT_CARD")
	}
	if ssnRegex.MatchString(text) {
		text = ssnRegex.ReplaceAllString(text, "[SSN_REDACTED]")
		types = append(types, "SSN")
	}
	if phoneRegex.MatchString(text) {
		text = phoneRegex.ReplaceAllString(text, "[PHONE_REDACTED]")
		types = append(types, "PHONE")
	}
	if apiKeyRegex.MatchString(text) {
		text = apiKeyRegex.ReplaceAllString(text, "[API_KEY_REDACTED]")
		types = append(types, "API_KEY")
	}

	return SanitizeReport{
		CleanedText:   text,
		RedactedTypes: types,
		WasSanitized:  original != text,
	}
}
