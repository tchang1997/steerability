$("#fileSelect").change(function () {
    const file = $(this).val();
    if (!file) return;
    $("#filename").val(file);
  
    $.post({
      url: "/columns",
      contentType: "application/json",
      data: JSON.stringify({ filename: file }),
      success: function (cols) {
        $("#xcol").empty();
        $("#ycol").empty();
        cols.forEach(col => {
          $("#xcol").append(`<option value="${col}">${col}</option>`);
          $("#ycol").append(`<option value="${col}">${col}</option>`);
        });
        if (cols.length >= 2) {
          $("#xcol").val(cols[0]);
          $("#ycol").val(cols[cols.length - 1]);  // pick the "last" one
        }
      }
    });
  });

let showSourcePoints = true;

$("#showSource").change(function () {
  showSourcePoints = this.checked;
  redraw();  // useful if in noLoop mode
});

$("#flip-axes-btn").click(function () {
  const x = $("#xcol").val();
  const y = $("#ycol").val();
  $("#xcol").val(y);
  $("#ycol").val(x);
});

function showStatus(msg, color = "red") {
  $("#status").text(msg).css("color", color);
}

function updateExportButtonState() {
  const btn = document.getElementById("export-btn");
  btn.disabled = !(flow_success && !capturing);
}


function cancelExport() {
  capturing = false;
  captureFrameCount = 0;
  capturer.stop();
  capturer = null;

  // Hide progress and reset button state
  document.getElementById("exportProgressWrapper").style.display = "none";
  document.getElementById("exportProgressBar").style.width = "0%";
  document.getElementById("exportProgressText").textContent = "";

  // Hide cancel button, re-enable export
  document.getElementById("cancelExportBtn").style.display = "none";
  updateExportButtonState?.();
  loop();
}

let running = false;
let flow_success = false;
let flow_title = "";
let particles = [];
const paddingLeft = 60;
const paddingRight = 20;
const paddingTop = 40;
const paddingBottom = 40;

function initializePlot() {
    background(15, 15, 20);
    drawAxes();  // gridlines, ticks, axis labels
}


let flowColors = [];
let canvasElt;
function setup() {
  textFont('Courier New');

  const canvas = createCanvas(800, 840); 
  canvas.parent("flow-canvas");
  canvasElt = canvas.elt;
  frameRate(60);

  noLoop(); // optional: remove if animating
  initializePlot();

  //【Ｉ　<３　ＶＡＰＯＲＷＡＶＥ】
  const stops = [  
    { stop: 0.0, color: lerpColor(color("#00C5F8"), color("#000000"), 0.2)}, // Faded sea-cyan
    { stop: 0.1, color: lerpColor(color("#11B4F5"), color("#000000"), 0.2)}, // Cerulean
    { stop: 0.25, color: "#4605EC" }, // Indigo
    { stop: 0.5, color: "#8705E4" }, // Purple
    { stop: 0.8, color: "#FF06C1" }, // Hot Pink
    { stop: 1.0, color: "#FF3366" }  // Red-ish Pink
  ];

  for (let i = 0; i <= 100; i++) {
    let normMag = i / 100.0;
    flowColors[i] = getColorFromStops(normMag, stops);
  }
}

function getColorFromStops(normMag, stops) {
  for (let i = 0; i < stops.length - 1; i++) {
    if (normMag >= stops[i].stop && normMag <= stops[i + 1].stop) {
      const t = map(normMag, stops[i].stop, stops[i + 1].stop, 0, 1);
      return lerpColor(color(stops[i].color), color(stops[i + 1].color), t);
    }
  }
  return color(stops[stops.length - 1].color);
}


const FLOW_SPEED = 0.012;

function compressedFade(t) {
  const raw = 0.5 * (1 - Math.cos(2 * Math.PI * t));
  return Math.pow(raw, 1.5);  // steeper falloff
}

let xAxisLabel = "";
let yAxisLabel = "";
function draw() {
    if (!running && !capturing) {
        background(15, 15, 20);
        drawAxes();
        return;
    }
    try {
      // blendMode(ADD);  // luminous trails
      noStroke();
      fill(15, 15, 20, 40);  // dark fade
      //blendMode(BLEND);
      rect(0, 0, width, height);

      drawAxes();
      textAlign(CENTER);
      textFont("monospace", 14);  // monospace
      fill(255);  // light gray
      noStroke();

      if (flow_success) {
        text(flow_title, width / 2, paddingTop - 20);    
      }

      colorMode(HSB, 360, 100, 100, 100);
      if (showSourcePoints && field && field.source_x && field.source_y) {
          stroke(255);
          noFill();
          strokeWeight(1);
          for (let i = 0; i < field.source_x.length; i++) {
            const px = map(field.source_x[i], 0, 1, paddingLeft, width - paddingRight);
            const py = map(1 - field.source_y[i], 0, 1, paddingTop, height - paddingBottom);
            ellipse(px, py, 4, 4);
          }
        }
        
      const travel = 0.1;
      for (let p of particles) {
        // Advance the particle's phase
        p.t = (p.t + FLOW_SPEED) % 1.0;
        const t1 = p.t % 1.0;
        const t2 = (p.t + 0.5) % 1.0;

        const a1 = compressedFade(t1);
        const a2 = compressedFade(t2);
        const norm = Math.max(Math.pow(a1, 1) + Math.pow(a2, 1), 1e-5);

        const alpha1 = a1 / norm;
        const alpha2 = a2 / norm;

        for (let [t, alpha] of [[t1, alpha1], [t2, alpha2]]) {
          const x = p.x0 - 0.5 * travel * p.u + travel * p.u * t;
          const y = p.y0 - 0.5 * travel * p.v + travel * p.v * t;
          if (x < 0 || x > 1 || y < 0 || y > 1) continue;

          const px = map(x, 0, 1, paddingLeft, width - paddingRight);
          const py = map(1 - y, 0, 1, paddingTop, height - paddingBottom);

          let idx = Math.floor(p.mag / maxMag * 100);
          const baseColor = flowColors[idx]
          baseColor.setAlpha(alpha * 255);

          fill(baseColor);
          noStroke();
          ellipse(px, py, 3, 3);
        }
      }

      colorMode(RGB, 255);  // reset mode
      if (capturing) {
        if (captureFrameCount === 0) {
          capturer.start();
          document.getElementById("exportProgressWrapper").style.display = "block";
          document.getElementById("exportProgressText").style.display = "block";
          document.getElementById("cancelExportBtn").style.display = "block";


        }
        requestAnimationFrame(draw);
        capturer.capture(canvasElt);

        captureFrameCount++;

        // update pbar
        const percent = (captureFrameCount / framesPerCycle) * 100;
        document.getElementById("exportProgressBar").style.width = `${percent}%`;
        document.getElementById("exportProgressText").textContent =
          `exporting frame ${captureFrameCount}/${framesPerCycle} (${percent.toFixed(2)}%)`;
      

        if (captureFrameCount >= framesPerCycle) {
          capturer.stop();
          capturer.save();
          capturing = false;
          updateExportButtonState();

          // reset progress bar
          document.getElementById("exportProgressWrapper").style.display = "none";
          document.getElementById("exportProgressText").style.display = "none";
          document.getElementById("cancelExportBtn").style.display = "none";


          document.getElementById("exportProgressBar").style.width = "0%";
          document.getElementById("exportProgressText").textContent = "";
        }
      }
    } catch (err) {
      console.error("Uncaught error in draw():", err);
      noLoop();
    }
}
      
function drawAxes() {
    textFont('Courier New'); 
    stroke(50);
    strokeWeight(0.5);
    fill(200);
    textSize(12);
    textAlign(CENTER, CENTER);
  
    // Gridlines + ticks
    for (let i = 0; i <= 10; i++) {
        const val = i / 10;
        const x = map(val, 0, 1, paddingLeft, width - paddingRight);
        const y = map(1 - val, 0, 1, paddingTop, height - paddingBottom);
      
        // Grid lines
        stroke(220);
        line(x, paddingTop, x, height - paddingBottom);          // vertical gridline
        line(paddingLeft, y, width - paddingRight, y);           // horizontal gridline
      
        // Tick labels
        stroke(0);
        fill(200);
        textAlign(CENTER, TOP);
        text(val.toFixed(1), x, height - paddingBottom + 6);      // X tick label
      
        textAlign(RIGHT, CENTER);
        text(val.toFixed(1), paddingLeft - 8, y);                 // Y tick label
      
        // Tick marks
        stroke(180);
        line(x, height - paddingBottom - 5, x, height - paddingBottom + 5);  // X tick
        line(paddingLeft - 5, y, paddingLeft + 5, y);                        // Y tick
    }
    
    // Axes
    stroke(120);
    strokeWeight(1);
    line(paddingLeft, paddingTop, paddingLeft, height - paddingBottom);              // y-axis line
    line(paddingLeft, height - paddingBottom, width - paddingRight, height - paddingBottom); // x-axis line
    
    // Axis labels
    textSize(16);
    textAlign(CENTER);
    text(plotted_xcol, (paddingLeft + width - paddingRight) / 2, height - paddingBottom + 30);

    // Y-axis label
    push();
    translate(paddingLeft - 45, height / 2);
    rotate(-HALF_PI);
    text(plotted_ycol, 0, 0);
    pop();
  }

function resetFlowCanvas() {
  // Clears particles but keeps grid and axis
  background(15, 15, 20);  // dark grid background
  drawAxes();              // custom function to redraw ticks/axes
  particles = [];          // clear animated flow particles
}


let plotted_xcol = ""
let plotted_ycol = ""
function generatePlot() {
    const xcol = $("#xcol").val();
    const ycol = $("#ycol").val();
    const filename = $("#fileSelect").val();
    if (!xcol || !ycol || !filename) {
      showStatus("Please select a file and both axes.");
      return;
    }
  
    showStatus("Generating flow...");

    flow_title = `file: ${filename}\nsubspace: (${xcol}, ${ycol})`;
    plotted_xcol = `${xcol} (specified)`
    plotted_ycol = `${ycol} (unspecified)`
    $.post({
      url: "/generate_flow",
      contentType: "application/json",
      data: JSON.stringify({ xcol, ycol, filename }),
      success: () => {
        fieldLoaded = false;
        loadJSON("/static/_field.json", f => {
            field = f;
            particles = [];
            let maxMag = 0;
            for (let i = 0; i < field.x.length; i++) {
            const u = field.u[i] * 3.5;
            const v = field.v[i] * 3.5;
            const mag = Math.sqrt(u * u + v * v);
            maxMag = Math.max(maxMag, mag);

            particles.push({
                x0: field.x[i],
                y0: field.y[i],
                u: u,
                v: v,
                mag: mag,
                t: 0.  // phase offset
            });
            }

            window.maxMag = maxMag;  // make accessible in draw()
            running = true;
            loop();
            showStatus("Flow loaded.", "green");
            flow_success = true;
            updateExportButtonState();
        });
      },
      error: () => {
        resetFlowCanvas();
        showStatus("Error generating flow.");
        flow_success = false;
        updateExportButtonState();
      }
    });
  }
  

function getExportFilename() {
  const rawName = $("#fileSelect").val().replace(/\.csv$/, "");
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  return `${rawName}_x__${plotted_xcol}_y__${plotted_ycol}_${timestamp}`;
}
  

let capturer;
let capturing = false;
let captureFrameCount = 0;
const framesPerCycle = Math.floor(1.0 / FLOW_SPEED);  // full loop
function startGifCapture() {
  capturer = new CCapture({
    format: 'webm',
    framerate: 60,
    name: getExportFilename(), 
  });
  loop();  // ensure draw is running

  capturing = true;
  updateExportButtonState();
  captureFrameCount = 0;
}