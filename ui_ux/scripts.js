function nextPage() {
    const currentUrl = window.location.href;
    if (currentUrl.includes("page1")) {
        window.location.href = "page2.html";
    } else if (currentUrl.includes("page2")) {
        window.location.href = "page3.html";
    }
}
