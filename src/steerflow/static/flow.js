function computeSummary(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const min = sorted[0];
  const max = sorted[sorted.length - 1];
  const q1 = sorted[Math.floor(sorted.length * 0.25)];
  const median = sorted[Math.floor(sorted.length * 0.5)];
  const q3 = sorted[Math.floor(sorted.length * 0.75)];
  const iqr = q3 - q1;
  const yTop = 1.1 * max;

  return {
    min: min.toFixed(3),
    q1: q1.toFixed(3),
    median: median.toFixed(3),
    q3: q3.toFixed(3),
    max: max.toFixed(3),
    iqr: iqr.toFixed(3),
    yTop: yTop
  };
}


const choicesInstances = {};  // global or higher-scope object
function enhanceFileSelectWithSearch() {
  selectIds = ["fileSelect", "xcol", "ycol"]
  for (let itemId of selectIds) {
    const select = document.getElementById(itemId);
    if (!select) {
      console.warn(itemId, "not found in DOM");
      return;
    }
    choicesInstances[itemId] = new Choices(document.getElementById(itemId), {
      searchEnabled: true,
      shouldSort: false,
      itemSelectText: "",
      noChoicesText: 'No choices available',
      position: 'auto',        
      classNames: {
        containerOuter: 'choices custom-choices'  
      }
    });
  }

}


$("#fileSelect").change(function () {
    const file = $(this).val();
    if (!file) return;
    $("#filename").val(file); 
    
    $("#fileStatusText")
    .text("Loading file...")
    .removeClass("done error");
    $("#fileSpinnerContainer").css("display", "flex");

    $.post({
      url: "/columns",
      contentType: "application/json",
      data: JSON.stringify({ filename: file }),
      success: function (cols) {
        const newChoices = cols.map(col => ({
          value: col,
          label: col,
          selected: false
        }));
        
        if (cols.length >= 2) {
          newChoices[0].selected = true;
          newChoices[cols.length - 1].selected = true; // both marked; safe for separate selects
        }
        
        choicesInstances["xcol"].setChoices(newChoices, 'value', 'label', true);
        choicesInstances["ycol"].setChoices(newChoices, 'value', 'label', true);
        
        // Then explicitly set the correct selection per select box
        choicesInstances["xcol"].setChoiceByValue(cols[0]);
        choicesInstances["ycol"].setChoiceByValue(cols[cols.length - 1]);
        $("#fileStatusText")
        .text("Done!")
        .addClass("done");
      },
      error: function () {
        $("#fileStatusText")
          .text("Failed to load.")
          .addClass("error");
      },
      complete: function () {
        // Optionally fade out after a short delay
        setTimeout(() => {
          $("#fileSpinnerContainer").css("display", "none");;
        }, 500);
        
      }
    });

    $.post({
      url: "/summary",
      contentType: "application/json",
      data: JSON.stringify({ filename: file }),
      success: function (data) {
        $("#summary-panel").html(data.summary_html).show();
      }
    });

    $.post({
      url: "/steerability_values",
      contentType: "application/json",
      data: JSON.stringify({ filename: file }),
      success: function (data) {
        const steerStats = computeSummary(data["steering_error"]);
        const miscalStats = computeSummary(data["miscalibration"]);
        const orthoStats = computeSummary(data["orthogonality"]);

        const x1Center = 0.125;  // domain: [0, 0.25]
        const x2Center = 0.505;  // domain: [0.38, 0.63]
        const x3Center = 0.88;   // domain: [0.76, 1.0]


        Plotly.newPlot("plotly-container", [
          {
            type: "violin",
            y: data["steering_error"],
            name: "Steering Error",
            xaxis: "x1",
            yaxis: "y1",
            line: { color: "#ffff99" },
            fillcolor: "rgba(255, 255, 153, 0.5)",
            box: { visible: true },
            meanline: { visible: false },
            opacity: 0.7,
            spanmode: "hard",
            points: true,
            hoverinfo: "text",
            hoveron: "all",
            hovertemplate:
            `Min: ${steerStats.min}<br>` +
            `Q1: ${steerStats.q1}<br>` +
            `Median: ${steerStats.median}<br>` +
            `Q3: ${steerStats.q3}<br>` +
            `Max: ${steerStats.max}<extra></extra>` 
          },
          {
            type: "violin",
            y: data["miscalibration"],
            name: "Miscalibration",
            xaxis: "x2",
            yaxis: "y2",
            line: { color: "#00ffff" },
            fillcolor: "rgba(0, 255, 255, 0.5)",
            box: { visible: true },
            meanline: { visible: false },
            opacity: 0.7,
            spanmode: "hard",
            points: true,
            hoverinfo: "text",
            hoveron: "all",
            hovertemplate:
            `Min: ${miscalStats.min}<br>` +
            `Q1: ${miscalStats.q1}<br>` +
            `Median: ${miscalStats.median}<br>` +
            `Q3: ${miscalStats.q3}<br>` +
            `Max: ${miscalStats.max}<extra></extra>`
          },
          {
            type: "violin",
            y: data["orthogonality"],
            name: "Orthogonality",
            xaxis: "x3",
            yaxis: "y3",
            line: { color: "#ff66cc" },
            fillcolor: "rgba(255, 102, 204, 0.5)",
            box: { visible: true },
            meanline: { visible: false },
            opacity: 0.7,
            spanmode: "hard",
            points: true,
            hoverinfo: "text",
            hoveron: "all",
            hovertemplate:
            `Min: ${orthoStats.min}<br>` +
            `Q1: ${orthoStats.q1}<br>` +
            `Median: ${orthoStats.median}<br>` +
            `Q3: ${orthoStats.q3}<br>` +
            `Max: ${orthoStats.max}<extra></extra>`
          },
          {
            type: "scatter",
            x: ["Steering Error"],
            y: [steerStats.yTop],
            text: [`${steerStats.median}<br>(${steerStats.iqr})`],
            mode: "text",
            textposition: "top center",
            textfont: {
              family: "monospace",
              size: 12,
              color: "#ffff99"
            },
            showlegend: false,
            hoverinfo: "skip",
            xaxis: "x1",
            yaxis: "y1",
          },
          {
            type: "scatter",
            x: ["Miscalibration"],
            y: [miscalStats.yTop],
            text: [`${miscalStats.median}<br>(${miscalStats.iqr})`],
            mode: "text",
            textposition: "top center",
            textfont: {
              family: "monospace",
              size: 12,
              color: "#00ffff"
            },
            showlegend: false,
            hoverinfo: "skip",
            xaxis: "x2",
            yaxis: "y2"
          },
          {
            type: "scatter",
            x: ["Orthogonality"],
            y: [orthoStats.yTop],
            text: [`${orthoStats.median}<br>(${orthoStats.iqr})`],
            mode: "text",
            textposition: "top center",
            textfont: {
              family: "monospace",
              size: 12,
              color: "#ff66cc"
            },
            showlegend: false,
            hoverinfo: "skip",
            xaxis: "x3",
            yaxis: "y3"
          }
          
        ], {
          grid: { rows: 1, columns: 3, pattern: "independent" },
          yaxis: { title: "Steering Error", gridcolor: "#333", range: [0, steerStats.yTop * 1.2]},
          yaxis2: { title: "Miscalibration", gridcolor: "#333", range: [0, miscalStats.yTop * 1.2] },
          yaxis3: { title: "Orthogonality", gridcolor: "#333", range: [0, orthoStats.yTop * 1.2] },
          xaxis: { domain: [0, 0.25], showticklabels: false, type: "category"},
          xaxis2: { domain: [0.38, 0.63], showticklabels: false, type: "category" },
          xaxis3: { domain: [0.76, 1], showticklabels: false, type: "category" },
          font: { color: "#eee", family: "monospace" },
          paper_bgcolor: "#181818",
          plot_bgcolor: "#181818",
          margin: { t: 10, b: 10 },
          legend: {
            orientation: "h",  
            yanchor: "bottom",
            y: -0.3,            
            xanchor: "center",
            x: 0.5,
            itemwidth: 30,
          }
        });
        
      }
    });
    
  });

let showSourcePoints = true;

$("#showSource").change(function () {
  showSourcePoints = this.checked;
  redraw();  // useful if in noLoop mode
});

$("#flip-axes-btn").click(function () {
  const x = choicesInstances["xcol"].getValue(true); // true = return value only
  const y = choicesInstances["ycol"].getValue(true);

  choicesInstances["xcol"].setChoiceByValue(y);
  choicesInstances["ycol"].setChoiceByValue(x);
});

function showStatus(msg, color = "red") {
  $("#status").text(msg).css("color", color);
}

function updateExportButtonState() {
  //const btn = document.getElementById("export-btn");
  //btn.disabled = !(flow_success && !capturing);

  disableIfCapturing = ["#flip-axes-btn", ".generate-btn", "#showSource", "#fileSelect", "#xcol", "#ycol"]
  disableIfCapturing.forEach(sel => {
    const el = document.querySelector(sel);
    if (el) el.disabled = capturing; 
  });

  disableIfCapturingOrFailed = ["#export-btn"]
  disableIfCapturingOrFailed.forEach(sel => {
    const el = document.querySelector(sel);
    if (el) el.disabled = !(flow_success && !capturing);
  }); 
}


function cancelExport() {
  capturing = false;
  captureFrameCount = 0;
  capturer.stop();
  capturer = null;

  // Hide progress and reset button state
  document.getElementById("cancelExportBtn").style.visibility = "hidden";
  document.getElementById("exportProgressBar").style.width = "0%";
  document.getElementById("exportProgressText").textContent = "no export in progress";

  // Hide cancel button, re-enable export
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

  const canvas = createCanvas(600, 630); 
  canvas.parent("flow-canvas");
  canvasElt = canvas.elt;
  frameRate(60);

  noLoop(); // optional: remove if animating
  initializePlot();

  //【Ｉ　<３　ＶＡＰＯＲＷＡＶＥ】
  const stops = [  
    { stop: 0.0, color: lerpColor(color("#00D5F8"), color("#000000"), 0.2)}, // Faded sea-cyan
    { stop: 0.05, color: lerpColor(color("#11B4F5"), color("#000000"), 0.0)}, // Cerulean
    { stop: 0.2, color: "#4605EC" }, // Indigo
    { stop: 0.4, color: "#8705E4" }, // Purple
    { stop: 0.7, color: "#FF06C1" }, // Hot Pink
    { stop: 1.0, color: "#FF0845" }  // Red-ish Pink
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
  return Math.pow(raw, 2);  // steeper falloff
}

let xAxisLabel = "";
let yAxisLabel = "";
let reset_switch = false;
function draw() {
    if (!running && !capturing) {
        background(15, 15, 20);
        drawAxes();
        return;
    }
    if (reset_switch) {
      background(0);
      reset_switch = false;
    }
    try {
      noStroke();
      fill(15, 15, 20, 5);  // dark fade
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
        const cnorm_pow = 1
        const norm = Math.max(Math.pow(a1, cnorm_pow) + Math.pow(a2, cnorm_pow), 1e-5);

        const alpha1 = a1 / norm;
        const alpha2 = a2 / norm;

        for (let [t, alpha] of [[t1, alpha1], [t2, alpha2]]) {
          const x = p.x0 - 0.5 * travel * p.u + travel * p.u * t;
          const y = p.y0 - 0.5 * travel * p.v + travel * p.v * t;
          if (x < 0 || x > 1 || y < 0 || y > 1) continue;

          const px = map(x, 0, 1, paddingLeft, width - paddingRight);
          const py = map(1 - y, 0, 1, paddingTop, height - paddingBottom);

          if (color_mode == "magnitude") {
            let idx = Math.floor(p.mag / maxMag * 100);
          } else { // side-effect
            let idx = Math.floor(p.mag === 0 ? 0 : Math.abs(p.v / maxMag) * 100); // 100 sin(theta)
          }
          const baseColor = flowColors[idx]
          baseColor.setAlpha(alpha * 255);

          fill(baseColor);
          noStroke();
          ellipse(px, py, 2, 2);
        }
      }

      colorMode(RGB, 255);  // reset mode
      if (capturing) {
        if (captureFrameCount === 0) {
          capturer.start();
          document.getElementById("cancelExportBtn").style.visibility = "visible";
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
          document.getElementById("cancelExportBtn").style.visibility = "hidden";


          document.getElementById("exportProgressBar").style.width = "0%";
          document.getElementById("exportProgressText").textContent = "no export in progress";
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
            const u = field.u[i] * 1.5;
            const v = field.v[i] * 1.5;
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

            flow_title = `file: ${filename}\nsubspace: (${xcol}, ${ycol})`;
            plotted_xcol = `${xcol} (specified)`
            plotted_ycol = `${ycol} (unspecified)`
            reset_switch = true;
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

document.addEventListener("DOMContentLoaded", enhanceFileSelectWithSearch);