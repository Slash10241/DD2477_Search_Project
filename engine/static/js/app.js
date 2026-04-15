console.log("app.js loaded");
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
});

document.addEventListener("click", (event) => {
	const btn = event.target.closest("[data-summary-llm]");
	if (!btn) return;

	console.log("Generate summary with top-k...");
	// later: call backend here
});