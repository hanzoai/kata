// Hanzo Kata Go Quickstart.
// Demonstrates autonomous coding agent failure triage.
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

type ChatRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	MaxTokens   int       `json:"max_tokens"`
	Temperature float64   `json:"temperature"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatResponse struct {
	Choices []struct {
		Message struct {
			Role             string `json:"role"`
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
		} `json:"message"`
	} `json:"choices"`
}

func main() {
	apiHost := os.Getenv("HANZO_API_HOST")
	if apiHost == "" {
		apiHost = "https://api.hanzo.ai"
	}
	apiKey := os.Getenv("HANZO_API_KEY")

	fmt.Printf("Connecting to Kata at %s...\n", apiHost)

	prompt := "Repo: hanzoai/server\nError: compilation error: undefined symbol 'InitWorkerPool'\nGenerate minimal patch."

	reqBody := ChatRequest{
		Model: "zen",
		Messages: []Message{
			{Role: "system", Content: "You are Hanzo Kata, an autonomous coding agent."},
			{Role: "user", Content: prompt},
		},
		MaxTokens:   512,
		Temperature: 0.0,
	}

	jsonBytes, err := json.Marshal(reqBody)
	if err != nil {
		panic(err)
	}

	url := fmt.Sprintf("%s/v1/chat/completions", strings.TrimRight(apiHost, "/"))
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonBytes))
	if err != nil {
		panic(err)
	}

	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", apiKey))
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()

	bodyBytes, _ := io.ReadAll(resp.Body)
	var chatResp ChatResponse
	_ = json.Unmarshal(bodyBytes, &chatResp)

	if len(chatResp.Choices) > 0 {
		msg := chatResp.Choices[0].Message
		patch := msg.Content
		if patch == "" {
			patch = msg.ReasoningContent
		}
		fmt.Printf("\n--- Kata Patch Generated ---\n%s\n", patch)
	}
}
