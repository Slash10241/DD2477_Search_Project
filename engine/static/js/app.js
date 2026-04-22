console.log("app.js loaded");

function getCookie(name) {
	let cookieValue = null;
	if (document.cookie && document.cookie !== "") {
		const cookies = document.cookie.split(";");
		for (const cookie of cookies) {
			const trimmed = cookie.trim();
			if (trimmed.startsWith(`${name}=`)) {
				cookieValue = decodeURIComponent(trimmed.substring(name.length + 1));
				break;
			}
		}
	}
	return cookieValue;
}

function escapeHtml(value) {
	return String(value)
		.replaceAll("&", "&amp;")
		.replaceAll("<", "&lt;")
		.replaceAll(">", "&gt;")
		.replaceAll('"', "&quot;")
		.replaceAll("'", "&#39;");
}

function renderHighlightedResults(results) {
	const cards = document.querySelectorAll(".result-card");

	results.forEach((result, index) => {
		const card = cards[index];
		if (!card) return;

		const snippet = card.querySelector("[data-result-snippet]");
		if (!snippet) return;

		const highlightedText = result.highlighted_text;
		if (typeof highlightedText === "string" && highlightedText.length > 0) {
			snippet.innerHTML = highlightedText;
		}
	});
}


let latestFeedbackState = null;

function getFeedbackLabel(score) {
	switch (score) {
		case 0:
			return "0 · Not relevant";
		case 1:
			return "1 · Mentioned";
		case 2:
			return "2 · Some detail";
		case 3:
			return "3 · Exhaustive";
		default:
			return "0 · Not relevant";
	}
}

function renderFeedbackResults(results) {
	const cards = document.querySelectorAll(".result-card");

	results.forEach((result, index) => {
		const card = cards[index];
		if (!card) return;

		const pill = card.querySelector("[data-feedback-pill]");
		if (!pill) return;

		const relevance = Number(result.feedback_relevance ?? 0);
		pill.hidden = false;
		pill.textContent = getFeedbackLabel(relevance);
		pill.setAttribute("data-feedback-score", String(relevance));
	});
}

function renderFeedbackMetrics(metrics) {
	const shell = document.querySelector("[data-feedback-metrics]");
	if (!shell) return;

	const precisionEl = shell.querySelector("[data-feedback-precision]");
	const mrrEl = shell.querySelector("[data-feedback-mrr]");
	const ndcgEl = shell.querySelector("[data-feedback-ndcg]");

	if (precisionEl) precisionEl.textContent = Number(metrics.precision_at_k ?? 0).toFixed(4);
	if (mrrEl) mrrEl.textContent = Number(metrics.mrr ?? 0).toFixed(4);
	if (ndcgEl) ndcgEl.textContent = Number(metrics.ndcg_at_k ?? 0).toFixed(4);

	shell.hidden = false;
}


function buildFeedbackExportText() {
	const queryInput = document.querySelector("input[name='q']");
	const cards = document.querySelectorAll(".result-card");

	const query = queryInput instanceof HTMLInputElement ? queryInput.value.trim() : "";
	const topK = latestFeedbackState?.topK ?? cards.length;

	const lines = [];
	lines.push(`${query} ${topK}`);
	lines.push("");

	cards.forEach((card, index) => {
		const pill = card.querySelector("[data-feedback-pill]");
		const score = pill ? Number(pill.getAttribute("data-feedback-score") || "0") : 0;

		const chunkText =
			card.querySelector("[data-result-snippet]")?.textContent?.trim() || "";

		lines.push(`${index}`);
		lines.push(chunkText);
		lines.push(`score ${score}`);
		lines.push("");
	});

	return lines.join("\n");
}

function downloadTextFile(filename, content) {
	const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
	const url = URL.createObjectURL(blob);

	const a = document.createElement("a");
	a.href = url;
	a.download = filename;
	document.body.appendChild(a);
	a.click();
	document.body.removeChild(a);

	URL.revokeObjectURL(url);
}


document.addEventListener("DOMContentLoaded", () => {
	const picker = document.querySelector("[data-mode-picker]");
	const sidebar = document.querySelector("[data-llm-sidebar]");
	const sidebarToggle = document.querySelector("[data-llm-sidebar-toggle]");
	const sidebarClose = document.querySelector("[data-llm-sidebar-close]");
	const modeButtons = document.querySelectorAll("[data-llm-mode]");
	const sections = document.querySelectorAll("[data-llm-section]");

	if (picker) {
		const trigger = picker.querySelector("[data-mode-trigger]");
		const menu = picker.querySelector("[data-mode-menu]");
		const label = picker.querySelector("[data-mode-label]");
		const hiddenModeInput = document.querySelector("#mode-input");
		const form = picker.closest("form");

		if (trigger && menu && label && hiddenModeInput && form) {
			const closeMenu = () => {
				picker.classList.remove("is-open");
				trigger.setAttribute("aria-expanded", "false");
			};

			const openMenu = () => {
				picker.classList.add("is-open");
				trigger.setAttribute("aria-expanded", "true");
			};

			const submitSearch = () => {
				if (typeof form.requestSubmit === "function") {
					form.requestSubmit();
					return;
				}

				form.dispatchEvent(
					new Event("submit", {
						bubbles: true,
						cancelable: true,
					}),
				);
			};

			trigger.addEventListener("click", () => {
				if (picker.classList.contains("is-open")) {
					closeMenu();
				} else {
					openMenu();
				}
			});

			menu.addEventListener("click", (event) => {
				const target = event.target instanceof Element ? event.target.closest("[data-mode-option]") : null;
				if (!target) {
					return;
				}

				const mode = target.getAttribute("data-mode-option");
				if (!mode) {
					return;
				}

				hiddenModeInput.value = mode;
				label.textContent = target.querySelector(".mode-picker-option-title")?.textContent ?? mode;

				menu.querySelectorAll("[data-mode-option]").forEach((option) => {
					option.classList.toggle("is-active", option.getAttribute("data-mode-option") === mode);
				});

				closeMenu();
				submitSearch();
			});

			document.addEventListener("click", (event) => {
				if (!picker.contains(event.target)) {
					closeMenu();
				}
			});

			document.addEventListener("keydown", (event) => {
				if (event.key === "Escape") {
					closeMenu();
				}
			});
		}
	}

	document.addEventListener("click", (event) => {
		const toggle = event.target instanceof Element ? event.target.closest("[data-result-expand-button]") : null;
		if (!toggle) {
			return;
		}

		const shell = toggle.closest("[data-result-snippet-shell]");
		if (!shell) {
			return;
		}

		const isExpanded = shell.classList.toggle("is-expanded");
		toggle.setAttribute("aria-expanded", String(isExpanded));
		toggle.textContent = isExpanded ? "Show less" : "Show more";
	});

	if (sidebar && sidebarToggle && sidebarClose) {
		const openSidebar = () => {
			sidebar.classList.add("is-open");
			sidebarToggle.setAttribute("aria-expanded", "true");
		};

		const closeSidebar = () => {
			sidebar.classList.remove("is-open");
			sidebarToggle.setAttribute("aria-expanded", "false");
		};

		sidebarToggle.addEventListener("click", () => {
			if (sidebar.classList.contains("is-open")) {
				closeSidebar();
			} else {
				openSidebar();
			}
		});

		sidebarClose.addEventListener("click", closeSidebar);

		document.addEventListener("keydown", (event) => {
			if (event.key === "Escape") {
				closeSidebar();
			}
		});

		document.addEventListener("click", (event) => {
			const target = event.target;
			if (!(target instanceof Element)) {
				return;
			}

			if (!sidebar.contains(target)) {
				closeSidebar();
			}
		});
	}

	if (modeButtons.length && sections.length) {
		modeButtons.forEach((button) => {
			button.addEventListener("click", () => {
				const selectedMode = button.getAttribute("data-llm-mode");

				modeButtons.forEach((item) => {
					item.classList.toggle("is-active", item === button);
				});

				sections.forEach((section) => {
					section.classList.toggle(
						"is-active",
						section.getAttribute("data-llm-section") === selectedMode,
					);
				});
			});
		});
	}

	document.addEventListener("click", async (event) => {
		const highlightButton = event.target instanceof Element
			? event.target.closest("[data-llm-highlight-button]")
			: null;

		if (!highlightButton) {
			return;
		}

		const queryInput = document.querySelector("input[name='q']");
		const modeInput = document.querySelector("#mode-input");
		const topKSelect = document.querySelector("#llm-highlight-k");

		const q = queryInput instanceof HTMLInputElement ? queryInput.value.trim() : "";
		const mode = modeInput instanceof HTMLInputElement ? modeInput.value : "lexical";
		const topK = topKSelect instanceof HTMLSelectElement ? Number(topKSelect.value) : 10;

		if (!q) {
			window.alert("Please enter a search query first.");
			return;
		}

		highlightButton.setAttribute("disabled", "true");
		const originalText = highlightButton.textContent;
		highlightButton.textContent = "Extracting...";

		try {
			const response = await fetch("/llm/highlight", {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
					"X-CSRFToken": getCookie("csrftoken") || "",
				},
				body: JSON.stringify({
					q,
					mode,
					top_k: topK,
				}),
			});

			const data = await response.json();

			if (!response.ok) {
				throw new Error(data.error || "Highlight extraction failed.");
			}

			renderHighlightedResults(data.results || []);
		} catch (error) {
			console.error(error);
			window.alert(error instanceof Error ? error.message : "Highlight extraction failed.");
		} finally {
			highlightButton.removeAttribute("disabled");
			highlightButton.textContent = originalText || "Extract Highlights";
		}
	});
});

// document.addEventListener("click", (event) => {
// 	const btn = event.target instanceof Element ? event.target.closest("[data-summary-llm]") : null;
// 	if (!btn) return;

// 	console.log("Generate summary with top-k...");
// 	// later: call backend here
// });

document.addEventListener("click", async (event) => {
  const btn = event.target.closest("[data-summary-llm]");
  if (!btn) return;

  console.log("Summarize clicked");

  const q = new URLSearchParams(window.location.search).get("q");
  const mode = new URLSearchParams(window.location.search).get("mode") || "lexical";

  try {
    const res = await fetch("/llm/summarize", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": getCookie("csrftoken") || "",
      },
      body: JSON.stringify({ q, mode, top_k: 8 }),
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Summary failed");
    }

    if (data.summary) {
      document.querySelector(".results-summary-text").textContent = data.summary;
    }

  } catch (err) {
    console.error(err);
    alert(err.message);
  }
});


document.addEventListener("click", async (event) => {
	const feedbackButton = event.target instanceof Element
		? event.target.closest("[data-llm-feedback-button]")
		: null;

	if (!feedbackButton) {
		return;
	}

	const queryInput = document.querySelector("input[name='q']");
	const modeInput = document.querySelector("#mode-input");
	const topKSelect = document.querySelector("#llm-feedback-k");

	const q = queryInput instanceof HTMLInputElement ? queryInput.value.trim() : "";
	const mode = modeInput instanceof HTMLInputElement ? modeInput.value : "lexical";
	const topK = topKSelect instanceof HTMLSelectElement ? Number(topKSelect.value) : 10;

	if (!q) {
		window.alert("Please enter a search query first.");
		return;
	}

	feedbackButton.setAttribute("disabled", "true");
	const originalText = feedbackButton.textContent;
	feedbackButton.textContent = "Scoring...";

	try {
		const response = await fetch("/llm/feedback", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				"X-CSRFToken": getCookie("csrftoken") || "",
			},
			body: JSON.stringify({
				q,
				mode,
				top_k: topK,
			}),
		});

		const data = await response.json();

		if (!response.ok) {
			throw new Error(data.error || "Feedback scoring failed.");
		}

		renderFeedbackResults(data.results || []);
		renderFeedbackMetrics(data.metrics || {});

		latestFeedbackState = {
			results: data.results || [],
			metrics: data.metrics || {},
			query: q,
			mode,
			topK,
		};
	} catch (error) {
		console.error(error);
		window.alert(error instanceof Error ? error.message : "Feedback scoring failed.");
	} finally {
		feedbackButton.removeAttribute("disabled");
		feedbackButton.textContent = originalText || "Score Results";
	}
});

document.addEventListener("click", (event) => {
	const saveButton = event.target instanceof Element
		? event.target.closest("[data-save-feedback-button]")
		: null;

	if (!saveButton) {
		return;
	}

	if (!latestFeedbackState) {
		window.alert("Please score the results first.");
		return;
	}

	const q = latestFeedbackState.query || "query";
	const safeName = q.replace(/[^\w\s-]/g, "").trim().replace(/\s+/g, "_") || "feedback";
	const content = buildFeedbackExportText();

	downloadTextFile(`${safeName}_feedback.txt`, content);
});



document.addEventListener("click", async (event) => {
	const askButton = event.target instanceof Element
		? event.target.closest("[data-llm-ask-button]")
		: null;

	if (!askButton) {
		return;
	}

	const searchQueryInput = document.querySelector("input[name='q']");
	const askQueryInput = document.querySelector("#llm-free-query");
	const modeInput = document.querySelector("#mode-input");
	const topKSelect = document.querySelector("#llm-top-k");

	const q = searchQueryInput instanceof HTMLInputElement ? searchQueryInput.value.trim() : "";
	const askQuery = askQueryInput instanceof HTMLTextAreaElement ? askQueryInput.value.trim() : "";
	const mode = modeInput instanceof HTMLInputElement ? modeInput.value : "lexical";
	const topK = topKSelect instanceof HTMLSelectElement ? Number(topKSelect.value) : 5;

	if (!q) {
		window.alert("Please enter a search query first.");
		return;
	}

	if (!askQuery) {
		window.alert("Please enter a question for the RAG query.");
		return;
	}

	askButton.setAttribute("disabled", "true");
	const originalText = askButton.textContent;
	askButton.textContent = "Running...";

try {
	const response = await fetch("/llm/ask", {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
			"X-CSRFToken": getCookie("csrftoken") || "",
		},
		body: JSON.stringify({
			q,
			ask_query: askQuery,
			mode,
			top_k: topK,
		}),
	});

	const data = await response.json();

	if (!response.ok) {
		throw new Error(data.error || "RAG query failed.");
	}

	renderAskAnswer(data.answer || "No answer returned.");
} catch (error) {
	console.error(error);
	renderAskAnswer(error instanceof Error ? error.message : "RAG query failed.");
} finally {
	askButton.removeAttribute("disabled");
	askButton.textContent = originalText || "Run RAG Query";
}
});

function renderAskAnswer(answer) {
	const answerBlock = document.querySelector("[data-llm-answer-block]");
	const answerText = document.querySelector("[data-llm-answer-text]");

	if (!answerBlock || !answerText) {
		return;
	}

	answerText.textContent = answer || "No answer returned.";
	answerBlock.hidden = false;
}