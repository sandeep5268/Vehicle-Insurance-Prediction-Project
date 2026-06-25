// ===========================
// OPEN FORM
// ===========================

function openForm() {

    document.querySelector(".hero-container").style.display = "none";
    document.getElementById("formModal").style.display = "flex";

}

// ===========================
// CLOSE FORM
// ===========================

function closeForm() {

    document.getElementById("formModal").style.display = "none";
    document.querySelector(".hero-container").style.display = "flex";

}

// ===========================
// BACK BUTTON
// ===========================

function goBack() {

    closeForm();

}

// ===========================
// RESULT POPUP
// ===========================

function goHome() {

    document.getElementById("popup").style.display = "none";
    document.querySelector(".hero-container").style.display = "flex";

}

// ===========================
// MOBILE TOOLTIP
// ===========================

document.querySelectorAll(".info-icon").forEach(icon => {

    icon.addEventListener("click", function (e) {

        e.stopPropagation();
        document.querySelectorAll(".info-icon").forEach(item =>{
            if(item != this){
                this.classList.remove("active")
            }
        });

        this.classList.toggle("active");

    });

});

document.addEventListener("click", () => {

    document.querySelectorAll(".info-icon").forEach(icon => {

        icon.classList.remove("active");

    });

});

window.onload = function () {

    if (typeof showPopup !== "undeifned" && showPopup) {

        document.querySelector(".hero-container").style.display = "none";

        document.getElementById("formModal").style.display = "none";

        const popup = document.getElementById("popup");
        if(popup){
            popup.style.display = "flex";
        }

    }

};