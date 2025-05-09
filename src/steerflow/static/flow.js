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


let running = false;
let particles = [];
const numParticles = 500;
const paddingLeft = 60;
const paddingRight = 20;
const paddingTop = 20;
const paddingBottom = 40;

function setup() {
    textFont('Courier New');
    const canvas = createCanvas(600, 600);
    canvas.parent("flow-canvas");

    for (let i = 0; i < numParticles; i++) {
        particles.push({ x: random(), y: random(), age: 0 });
    }

    frameRate(60);
    noLoop();
    initializePlot();
}
  
function initializePlot() {
    background(15, 15, 20);
    drawAxes();  // gridlines, ticks, axis labels
}

function getJetColor(normMag) {
    normMag = constrain(normMag, 0, 1);


    // Hue 260 (deep blue) → 0 (red), full sat/bright
    const hue = map(normMag, 0, 1, 260, 345);
    colorMode(HSB, 360, 100, 100, 100);
    const col = color(hue, 100, 100, 100);  // Full neon
    colorMode(RGB, 255);
    return col;
}



let xAxisLabel = "";
let yAxisLabel = "";
function draw() {
    if (!running) {
        background(15, 15, 20);
        drawAxes();
        return;
    }

    blendMode(ADD);  // luminous trails
    noStroke();
    fill(15, 15, 20, 10);  // dark fade
    blendMode(BLEND);
    rect(0, 0, width, height);

    drawAxes();
    textAlign(CENTER);
    textFont("monospace", 16);  // monospace
    fill(255);  // light gray
    noStroke();
    xAxisLabel = $("#xcol").val();
    yAxisLabel = $("#ycol").val();
    text(`subspace: (${xAxisLabel}, ${yAxisLabel})`, width / 2, paddingTop - 10);

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

      for (let p of particles) {
        p.t += 0.01;
        const baseTravel = 0.1
        const travel = baseTravel * p.mag / maxMag;  // longer streaks for faster flow     
        const progress = p.t % 1;

        const base_x = p.x0 - 0.5 * travel * p.u;
        const base_y = p.y0 - 0.5 * travel * p.v;

        const x = base_x + travel * p.u * progress;
        const y = base_y + travel * p.v * progress;

        if (x < 0 || x > 1 || y < 0 || y > 1) continue;

        const px = map(x, 0, 1, paddingLeft, width - paddingRight);
        const py = map(1 - y, 0, 1, paddingTop, height - paddingBottom);

        const col = getJetColor(p.mag / maxMag);
        fill(col);
        noStroke();
        ellipse(px, py, 2.5, 2.5);
        
    }
    
    colorMode(RGB, 255);  // reset mode
}
      


function drawArrow(base, vec) {
  push();
  translate(base.x, base.y);
  line(0, 0, vec.x, vec.y);
  rotate(vec.heading());
  const arrowSize = 6;
  translate(vec.mag() - arrowSize, 0);
  triangle(0, arrowSize / 2, 0, -arrowSize / 2, arrowSize, 0);
  pop();
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
    text("X", (paddingLeft + width - paddingRight) / 2, height - 5);
    push();
    translate(paddingLeft - 45, height / 2);
    rotate(-HALF_PI);
    text("Y", 0, 0);
    pop();
  }
  
function showStatus(msg, color = "red") {
    $("#status").text(msg).css("color", color);
}
  
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
        });
      },
      error: () => showStatus("Error generating flow.")
    });
  }
  