/*
 * Colour-stop editor for the NDDG "Interactive Gradient" tab.
 *
 * The ComfyUI node carried its own canvas widget on the litegraph node body.  There is no
 * equivalent surface in a WebUI tab, so this mounts a canvas next to the JSON textbox and
 * keeps the two in sync: the canvas is the editor, the textbox stays the single source of
 * truth that the Python side reads.
 *
 *   click empty space    add a stop at that position, in the current colour
 *   drag a stop          move it
 *   click a stop         select it (the colour picker then edits that stop)
 *   right-click a stop   remove it (the last stop cannot be removed)
 */

(function () {
    "use strict";

    const MOUNT_ID = "nddg_gradient_canvas";
    const TEXTBOX_ID = "nddg_gradient_data";
    const HIT_RADIUS = 11;

    let selected = 0;
    let dragging = -1;

    function textarea() {
        const root = gradioApp().querySelector("#" + TEXTBOX_ID);
        return root ? root.querySelector("textarea, input") : null;
    }

    function readStops() {
        const field = textarea();
        if (!field) return [];
        try {
            const parsed = JSON.parse(field.value);
            if (!Array.isArray(parsed)) return [];
            return parsed.filter(function (s) {
                return s && typeof s.color === "string" && isFinite(s.x) && isFinite(s.y);
            });
        } catch (e) {
            return [];
        }
    }

    function writeStops(stops) {
        const field = textarea();
        if (!field) return;
        field.value = JSON.stringify(stops.map(function (s) {
            return {x: Math.round(s.x * 1000) / 1000, y: Math.round(s.y * 1000) / 1000, color: s.color};
        }));
        // Gradio 4 listens on "input"; without this the Python side never sees the change.
        field.dispatchEvent(new Event("input", {bubbles: true}));
    }

    function build(mount) {
        mount.innerHTML = "";
        mount.style.display = "flex";
        mount.style.flexDirection = "column";
        mount.style.gap = "6px";

        const canvas = document.createElement("canvas");
        canvas.width = 480;
        canvas.height = 270;
        canvas.style.width = "100%";
        canvas.style.maxWidth = "480px";
        canvas.style.height = "auto";
        canvas.style.cursor = "crosshair";
        canvas.style.borderRadius = "6px";
        canvas.style.border = "1px solid var(--border-color-primary, #444)";
        canvas.style.background = "#1b1b1b";
        canvas.title = "click to add · drag to move · right-click to remove";

        const controls = document.createElement("div");
        controls.style.display = "flex";
        controls.style.alignItems = "center";
        controls.style.gap = "8px";
        controls.style.flexWrap = "wrap";

        const picker = document.createElement("input");
        picker.type = "color";
        picker.value = "#ff3300";
        picker.style.width = "48px";
        picker.style.height = "28px";
        picker.style.padding = "0";
        picker.style.border = "none";
        picker.style.background = "transparent";
        picker.title = "colour of the selected stop, and of the next one added";

        const label = document.createElement("span");
        label.style.fontSize = "0.8em";
        label.style.opacity = "0.75";

        const shuffle = makeButton("Randomise colours");
        const reset = makeButton("Reset");

        controls.appendChild(picker);
        controls.appendChild(label);
        controls.appendChild(shuffle);
        controls.appendChild(reset);

        mount.appendChild(canvas);
        mount.appendChild(controls);

        const context = canvas.getContext("2d");

        function draw() {
            const stops = readStops();
            if (selected >= stops.length) selected = Math.max(0, stops.length - 1);

            context.clearRect(0, 0, canvas.width, canvas.height);

            // cheap inverse-distance preview, at low resolution, purely as a guide
            const cell = 10;
            for (let py = 0; py < canvas.height; py += cell) {
                for (let px = 0; px < canvas.width; px += cell) {
                    context.fillStyle = mix(stops, (px + cell / 2) / canvas.width, (py + cell / 2) / canvas.height);
                    context.fillRect(px, py, cell, cell);
                }
            }

            stops.forEach(function (stop, index) {
                const px = stop.x * canvas.width;
                const py = stop.y * canvas.height;
                context.beginPath();
                context.arc(px, py, index === selected ? 10 : 8, 0, Math.PI * 2);
                context.fillStyle = stop.color;
                context.fill();
                context.lineWidth = index === selected ? 3 : 2;
                context.strokeStyle = index === selected ? "#ffffff" : "rgba(0,0,0,0.65)";
                context.stroke();
            });

            label.textContent = stops.length + (stops.length === 1 ? " stop" : " stops");
            if (stops[selected]) picker.value = stops[selected].color;
        }

        function positionOf(event) {
            const rect = canvas.getBoundingClientRect();
            return {
                x: Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)),
                y: Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height)),
                scale: canvas.width / rect.width,
            };
        }

        function hit(stops, point) {
            for (let i = stops.length - 1; i >= 0; i--) {
                const dx = (stops[i].x - point.x) * canvas.width;
                const dy = (stops[i].y - point.y) * canvas.height;
                if (Math.sqrt(dx * dx + dy * dy) <= HIT_RADIUS) return i;
            }
            return -1;
        }

        canvas.addEventListener("mousedown", function (event) {
            if (event.button !== 0) return;
            const stops = readStops();
            const point = positionOf(event);
            const index = hit(stops, point);

            if (index >= 0) {
                selected = index;
                dragging = index;
                draw();
                return;
            }

            stops.push({x: point.x, y: point.y, color: picker.value});
            selected = stops.length - 1;
            writeStops(stops);
            draw();
        });

        canvas.addEventListener("mousemove", function (event) {
            if (dragging < 0) return;
            const stops = readStops();
            if (!stops[dragging]) {
                dragging = -1;
                return;
            }
            const point = positionOf(event);
            stops[dragging].x = point.x;
            stops[dragging].y = point.y;
            writeStops(stops);
            draw();
        });

        ["mouseup", "mouseleave"].forEach(function (name) {
            canvas.addEventListener(name, function () {
                dragging = -1;
            });
        });

        canvas.addEventListener("contextmenu", function (event) {
            event.preventDefault();
            const stops = readStops();
            if (stops.length <= 1) return;
            const index = hit(stops, positionOf(event));
            if (index < 0) return;
            stops.splice(index, 1);
            selected = Math.min(selected, stops.length - 1);
            writeStops(stops);
            draw();
        });

        picker.addEventListener("input", function () {
            const stops = readStops();
            if (!stops[selected]) return;
            stops[selected].color = picker.value;
            writeStops(stops);
            draw();
        });

        shuffle.addEventListener("click", function () {
            const stops = readStops().map(function (stop) {
                return {x: stop.x, y: stop.y, color: randomHex()};
            });
            writeStops(stops);
            draw();
        });

        reset.addEventListener("click", function () {
            selected = 0;
            writeStops([
                {x: 0.2, y: 0.5, color: "#ff3300"},
                {x: 0.8, y: 0.5, color: "#00ffe1"},
            ]);
            draw();
        });

        const field = textarea();
        if (field) field.addEventListener("change", draw);

        draw();
    }

    function makeButton(text) {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = text;
        button.className = "lg secondary gradio-button";
        button.style.padding = "2px 10px";
        button.style.fontSize = "0.8em";
        button.style.minWidth = "0";
        return button;
    }

    function randomHex() {
        return "#" + Math.floor(Math.random() * 0xffffff).toString(16).padStart(6, "0");
    }

    function mix(stops, x, y) {
        if (!stops.length) return "#1b1b1b";
        let r = 0, g = 0, b = 0, total = 0;
        for (const stop of stops) {
            const dx = x - stop.x;
            const dy = y - stop.y;
            const weight = 1 / (dx * dx + dy * dy + 0.004);
            const value = parseInt(stop.color.slice(1), 16);
            r += ((value >> 16) & 255) * weight;
            g += ((value >> 8) & 255) * weight;
            b += (value & 255) * weight;
            total += weight;
        }
        return "rgb(" + Math.round(r / total) + "," + Math.round(g / total) + "," + Math.round(b / total) + ")";
    }

    onUiLoaded(function () {
        const mount = gradioApp().querySelector("#" + MOUNT_ID);
        if (mount && !mount.dataset.nddgReady) {
            mount.dataset.nddgReady = "1";
            build(mount);
        }
    });
})();
