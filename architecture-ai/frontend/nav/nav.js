(function loadCSS() {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "../nav/nav.css";
    document.head.appendChild(link);
})();

function createNav() {
    const currentPage = window.location.pathname.split("/").pop() || "";
    const nav = document.createElement("nav");
    nav.className = "main-nav";

    nav.innerHTML = `
        <div class="nav-container">
            <a href="../documents/upload.html" class="nav-btn ${currentPage === 'upload.html' ? 'active' : ''}">
                Téléverser des documents
            </a>
            <a href="../documents/documents.html" class="nav-btn ${currentPage === 'documents.html' ? 'active' : ''}">
                Voir les documents
            </a>
            <a href="../projects/projects.html" class="nav-btn ${currentPage === 'projects.html' ? 'active' : ''}">
                Projets
            </a>
            <a href="../rc/rc-analyze.html" class="nav-btn ${currentPage === 'rc-analyze.html' ? 'active' : ''}">
                Analyse RC
            </a>
            <a href="../memoire/memoire-generation.html" class="nav-btn ${currentPage === 'memoire-generation.html' ? 'active' : ''}">
                Mémoire
            </a>
        </div>
    `;

    document.body.insertBefore(nav, document.body.firstChild);
}

document.addEventListener("DOMContentLoaded", createNav);