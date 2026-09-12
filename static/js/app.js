// Global small helpers shared across pages.
// Page-specific behaviour (camera control, capture, training) lives in
// inline <script> blocks in the relevant templates so it has direct
// access to server-rendered context (student IDs, folder names, etc.).

document.addEventListener("DOMContentLoaded", () => {
    // Auto-dismiss flash alerts after a few seconds.
    document.querySelectorAll(".alert").forEach((alertEl) => {
        setTimeout(() => {
            if (window.bootstrap) {
                const alert = window.bootstrap.Alert.getOrCreateInstance(alertEl);
                alert.close();
            }
        }, 6000);
    });
});
