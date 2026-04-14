document.addEventListener("DOMContentLoaded", () => {
	const picker = document.querySelector("[data-mode-picker]");
	if (!picker) {
		return;
	}

	const trigger = picker.querySelector("[data-mode-trigger]");
	const menu = picker.querySelector("[data-mode-menu]");
	const label = picker.querySelector("[data-mode-label]");
	const hiddenModeInput = document.querySelector("#mode-input");
	const form = picker.closest("form");

	if (!trigger || !menu || !label || !hiddenModeInput || !form) {
		return;
	}

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
});
