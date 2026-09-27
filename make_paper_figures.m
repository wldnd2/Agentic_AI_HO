%% make_paper_figures.m  (IEEE JSAC/TWC style)
% Journal-style figures for the CHO/PDCP agentic manuscript.
%
% Output:
%   <ROOT>/fig/png/figXX_*.png  (600 dpi)
%   <ROOT>/fig/pdf/figXX_*.pdf  (vector)
%
% Figures 01-15: evaluation results (single plot per figure, no error bars;
%                95% CIs are reported in the manuscript tables).
% Figure 16    : training convergence of the COMMON reward
%                r = -C_def - lambda*C_ctrl (mean over 5 training seeds).
%                The common reward is used for every model, including M5,
%                whose logged training reward additionally contains the
%                Lagrangian penalty mu*(C_out - eps) and is therefore not
%                comparable across models.
%
% Design principles:
%   - IEEE single-column width (3.5 in), Times New Roman 9 pt
%   - no in-figure titles, no value labels
%   - one accent colour for M5, grey tones for the other methods
%   - light dotted grid, open top/right borders, vector PDF export

clear; close all;

%% ========================= User setting ================================
ROOT = '/Users/jeonjiwoong/Desktop/CHO_simul';
%% ======================================================================

RAW    = fullfile(ROOT, 'outputs', 'unified_eval', 'raw');
FIGDIR = fullfile(ROOT, 'fig');
LAMBDA = 0.25;                      % reward weight lambda (Table I)

if ~exist(fullfile(FIGDIR, 'png'), 'dir'), mkdir(fullfile(FIGDIR, 'png')); end
if ~exist(fullfile(FIGDIR, 'pdf'), 'dir'), mkdir(fullfile(FIGDIR, 'pdf')); end

% IEEE single-column width. Heights are selected per plot below.
W  = 3.5;
H5 = 2.05;   % about 4-5 rows
H6 = 2.25;   % about 6 rows
H7 = 2.45;   % about 7 rows
H9 = 2.85;   % about 9 rows
HS = 2.35;   % scatter / trade-off
HT = 2.40;   % training curves

LEARNED = {'M1', 'M2', 'M3', 'M4', 'M5'};
ALLM    = [LEARNED, {'CHO-A3-TTT', 'Max-Observed-SINR', 'Always-Dup', 'Threshold-Dup'}];
ALLL    = [LEARNED, {'CHO (A3/TTT)', 'Max-SINR', 'Always-Dup', 'Threshold-Dup'}];
LCHO    = [LEARNED, {'CHO-A3-TTT'}];
LCHOL   = [LEARNED, {'CHO (A3/TTT)'}];
SEEDS   = [7 8 9 10 11];
EVALS   = [7 8 9 10 11];

%% ========================== Journal style ==============================
C.accent      = [0.000 0.365 0.635];  % M5
C.accentDark  = [0.000 0.245 0.430];
C.learn       = [0.760 0.815 0.865];
C.rule        = [0.690 0.690 0.690];
C.text        = [0.120 0.120 0.120];
C.axis        = [0.180 0.180 0.180];
C.grid        = [0.865 0.865 0.865];
C.separator   = [0.740 0.740 0.740];
C.zero        = [0.260 0.260 0.260];
C.positive    = [0.760 0.300 0.220];
C.negative    = C.accent;
C.white       = [1 1 1];
% training-curve lines for M1-M4 (grey levels + line styles, grayscale-safe)
C.curveGray   = [0.55 0.55 0.55; 0.40 0.40 0.40; 0.70 0.70 0.70; 0.25 0.25 0.25];
C.curveStyle  = {'--', ':', '-.', '-'};

S.fontName    = 'Times New Roman';
S.fontSize    = 9;
S.axisLW      = 0.65;
S.lineLW      = 1.15;
S.gridLW      = 0.45;
S.barHeight   = 0.58;
S.markerSize  = 6.2;
S.visible     = 'off';

set(groot, ...
    'DefaultFigureColor', 'w', ...
    'DefaultAxesFontName', S.fontName, ...
    'DefaultTextFontName', S.fontName, ...
    'DefaultAxesFontSize', S.fontSize, ...
    'DefaultTextFontSize', S.fontSize, ...
    'DefaultAxesLineWidth', S.axisLW, ...
    'DefaultLineLineWidth', S.lineLW, ...
    'DefaultAxesBox', 'off', ...
    'DefaultAxesTickDir', 'out', ...
    'DefaultAxesTickLength', [0.012 0.012], ...
    'DefaultAxesXColor', C.axis, ...
    'DefaultAxesYColor', C.axis, ...
    'DefaultAxesTickLabelInterpreter', 'tex', ...
    'DefaultTextInterpreter', 'tex', ...
    'DefaultLegendInterpreter', 'tex');

%% ======================== Load raw results =============================
files = dir(fullfile(RAW, '*.json'));
if isempty(files)
    error('No JSON result files found in: %s', RAW);
end

fprintf('Reading %d result files ...\n', numel(files));
R = struct('label', {}, 'cond', {}, 'ts', {}, 'es', {}, 'm', {});

for k = 1:numel(files)
    j = jsondecode(fileread(fullfile(files(k).folder, files(k).name)));

    label = j.model;
    if isfield(j, 'ablation') && ~strcmp(j.ablation, 'none')
        label = [j.model '[' j.ablation ']'];
    end

    m = j.metrics;
    m.delivery_ratio = 100 * m.delivered_packets / m.arrivals;
    m.overflow_ratio = 100 * m.overflow_packets / m.arrivals;

    R(end + 1) = struct( ...
        'label', label, ...
        'cond', j.condition, ...
        'ts', j.train_seed, ...
        'es', j.eval_seed, ...
        'm', m); %#ok<SAGROW>
end

M = @(label, cond, metric) mean(unit_values(R, label, cond, metric, SEEDS, EVALS));
FAILED = {};

%% ================= 01-02: outage ratio, all methods ====================
try
    journal_bar(M, ALLM, ALLL, 'nominal', 'outage_ratio', 1, ...
        'Outage ratio', C, S, 'fig01_outage_nominal', W, H9, FIGDIR);

    journal_bar(M, ALLM, ALLL, 'stress', 'outage_ratio', 1, ...
        'Outage ratio', C, S, 'fig02_outage_stress', W, H9, FIGDIR);
catch err
    FAILED = record_failure(FAILED, '01-02: outage ratio', err);
end

%% ================= 03-06: mobility efficiency ==========================
try
    journal_bar(M, LCHO, LCHOL, 'nominal', 'handover_started', 1, ...
        'HO executions per epoch', C, S, 'fig03_ho_executions', W, H6, FIGDIR);

    journal_bar(M, [LCHO, {'Max-Observed-SINR'}], [LCHOL, {'Max-SINR'}], ...
        'nominal', 'pingpong_rate', 1, 'Ping-pong ratio', ...
        C, S, 'fig04_pingpong', W, H7, FIGDIR);

    journal_bar(M, LCHO, LCHOL, 'nominal', 'handover_failure', 1, ...
        'Failed HOs per epoch', C, S, 'fig05_ho_failures', W, H6, FIGDIR);

    journal_bar(M, LEARNED, LEARNED, 'nominal', 'resource_cost', 1e3, ...
        'Resource cost (\times10^{-3})', C, S, ...
        'fig06_resource_learned', W, H5, FIGDIR);
catch err
    FAILED = record_failure(FAILED, '03-06: mobility efficiency', err);
end

%% ================= 07-08: delivered service ============================
try
    journal_bar(M, ALLM, ALLL, 'nominal', 'delivery_ratio', 1, ...
        'Packet delivery ratio (%)', C, S, ...
        'fig07_delivery_ratio', W, H9, FIGDIR);

    journal_bar(M, ALLM, ALLL, 'nominal', 'deadline_violation_rate', 100, ...
        'Deadline-violation rate (%)', C, S, ...
        'fig08_deadline_violation', W, H9, FIGDIR);
catch err
    FAILED = record_failure(FAILED, '07-08: delivered service', err);
end

%% ================= 09-11: CHO usage ===================================
try
    journal_bar(M, LCHO, LCHOL, 'nominal', 'prepare_to_commit_rate', 100, ...
        'Preparations ending in a switch (%)', C, S, ...
        'fig09_prep_to_commit', W, H6, FIGDIR);

    journal_bar(M, LCHO, LCHOL, 'nominal', 'preparation_rejection_rate', 100, ...
        'Rejected preparation requests (%)', C, S, ...
        'fig10_prep_rejection', W, H6, FIGDIR);

    journal_bar(M, LCHO, LCHOL, 'nominal', 'mean_prepared_duration', 1, ...
        'Preparation holding time (epochs)', C, S, ...
        'fig11_prep_holding_time', W, H6, FIGDIR);
catch err
    FAILED = record_failure(FAILED, '09-11: CHO usage', err);
end

%% ================= 12: M5 relative to M1 ===============================
try
    rm = {'pingpong_rate', 'handover_started', 'resource_cost', ...
          'handover_failure', 'outage_ratio'};
    rl = {'Ping-pong ratio', 'HO executions', 'Resource cost', ...
          'HO failures', 'Outage ratio'};

    v = zeros(1, numel(rm));
    for q = 1:numel(rm)
        v(q) = pct_change(M('M5', 'nominal', rm{q}), ...
                          M('M1', 'nominal', rm{q}), rm{q});
    end

    journal_relative(rl, v, 'Relative change of M5 vs. M1 (%)', ...
        C, S, 'fig12_m5_vs_m1', W, H5, FIGDIR);
catch err
    FAILED = record_failure(FAILED, '12: M5 relative to M1', err);
end

%% ================= 13: M5 relative to CHO ===============================
try
    rm = {'pingpong_rate', 'handover_failure', 'outage_ratio', ...
          'overflow_ratio', 'service_deficit', 'deadline_violation_rate'};
    rl = {'Ping-pong ratio', 'HO failures', 'Outage ratio', ...
          'Queue overflow', 'Service deficit', 'Deadline violations'};

    v = zeros(1, numel(rm));
    for q = 1:numel(rm)
        v(q) = pct_change(M('M5', 'nominal', rm{q}), ...
                          M('CHO-A3-TTT', 'nominal', rm{q}), rm{q});
    end

    journal_relative(rl, v, 'Relative change of M5 vs. CHO (%)', ...
        C, S, 'fig13_m5_vs_cho', W, H6, FIGDIR);
catch err
    FAILED = record_failure(FAILED, '13: M5 relative to CHO', err);
end

%% ================= 14: HO vs outage trade-off ==========================
try
    xs = zeros(1, numel(LEARNED));
    ys = zeros(1, numel(LEARNED));

    for k = 1:numel(LEARNED)
        xs(k) = M(LEARNED{k}, 'nominal', 'handover_started');
        ys(k) = M(LEARNED{k}, 'nominal', 'outage_ratio');
    end

    if any(~isfinite(xs)) || any(~isfinite(ys))
        error('Missing/non-finite value in trade-off figure.');
    end

    fh = new_figure(W, HS, S);
    ax = axes('Parent', fh);
    hold(ax, 'on');
    style_axes(ax, C, S, true);

    [xlo, xhi] = padded_limits(xs, 0.12);
    [ylo, yhi] = padded_limits(ys, 0.16);
    xlim(ax, [xlo xhi]);
    ylim(ax, [ylo yhi]);

    markers = {'o', 's', '^', 'd'};
    for k = 1:4
        plot(ax, xs(k), ys(k), markers{k}, ...
            'LineStyle', 'none', ...
            'MarkerSize', S.markerSize, ...
            'MarkerFaceColor', C.white, ...
            'MarkerEdgeColor', [0.38 0.38 0.38], ...
            'LineWidth', 0.9);
    end

    plot(ax, xs(5), ys(5), 'o', ...
        'LineStyle', 'none', ...
        'MarkerSize', S.markerSize + 1.3, ...
        'MarkerFaceColor', C.accent, ...
        'MarkerEdgeColor', C.accentDark, ...
        'LineWidth', 0.9);

    dx = 0.025 * (xhi - xlo);
    dy = 0.010 * (yhi - ylo);
    for k = 1:4
        text(ax, xs(k) + dx, ys(k) + dy, LEARNED{k}, ...
            'Color', C.text, ...
            'FontSize', S.fontSize - 0.5, ...
            'VerticalAlignment', 'middle');
    end
    text(ax, xs(5) + dx, ys(5) + dy, 'M5', ...
        'Color', C.accentDark, ...
        'FontWeight', 'bold', ...
        'FontSize', S.fontSize, ...
        'VerticalAlignment', 'middle');

    xlabel(ax, 'HO executions per epoch');
    ylabel(ax, 'Outage ratio');

    save_journal_fig(fh, 'fig14_tradeoff_learned', W, HS, FIGDIR);
catch err
    FAILED = record_failure(FAILED, '14: trade-off', err);
end

%% ================= 15: price ablation ==================================
try
    rm = {'handover_started', 'pingpong_rate', 'resource_cost', 'outage_ratio'};
    rl = {'HO executions', 'Ping-pong ratio', 'Resource cost', 'Outage ratio'};
    v = zeros(1, numel(rm));

    for q = 1:numel(rm)
        base = unit_values(R, 'M5', 'nominal', rm{q}, SEEDS, EVALS);
        abl  = unit_values(R, 'M5[price_zero]', 'nominal', rm{q}, SEEDS, EVALS);
        n = min(numel(base), numel(abl));

        if n == 0
            error('No paired values for price ablation metric: %s', rm{q});
        end
        if any(base(1:n) == 0)
            error('Zero denominator in price ablation metric: %s', rm{q});
        end

        v(q) = mean(100 * (abl(1:n) - base(1:n)) ./ base(1:n));
    end

    journal_relative(rl, v, 'Change after removing the price term (%)', ...
        C, S, 'fig15_m5_price_removed', W, H5, FIGDIR);
catch err
    FAILED = record_failure(FAILED, '15: price ablation', err);
end

%% ================= 16: training convergence (common reward) ============
% Mean over the five training seeds of the 50-iteration moving average of
% r = -C_def - lambda*C_ctrl, sampled every 10 iterations. No spread band.
try
    fh = new_figure(W, HT, S);
    ax = axes('Parent', fh);
    hold(ax, 'on');
    style_axes(ax, C, S, true);

    hl = zeros(1, numel(LEARNED));
    x = [];
    for k = 1:numel(LEARNED)
        Y = [];
        for s = SEEDS
            f = training_csv(ROOT, LEARNED{k}, s);
            if ~exist(f, 'file')
                warning('Missing training log: %s', f);
                continue;
            end
            [hdr, D] = read_numeric_csv(f);
            sdef = D(:, strcmp(hdr, 'service_deficit'));
            rc   = D(:, strcmp(hdr, 'resource_cost'));
            it   = D(:, strcmp(hdr, 'iteration'));
            if isempty(sdef) || isempty(rc) || isempty(it)
                warning('Required columns missing in %s; skipped.', f);
                continue;
            end
            y = moving_average(-sdef - LAMBDA * rc, 50);
            idx = 10:10:numel(y);
            Y = [Y; y(idx)']; %#ok<AGROW>
            x = it(idx)';
        end
        if isempty(Y)
            error('No training logs found for %s (checked %s)', LEARNED{k}, ...
                  fileparts(training_csv(ROOT, LEARNED{k}, SEEDS(1))));
        end
        fprintf('  %s: %d training logs used\n', LEARNED{k}, size(Y, 1));
        mu_k = mean(Y, 1);
        if strcmp(LEARNED{k}, 'M5')
            hl(k) = plot(ax, x, mu_k, '-', 'Color', C.accent, 'LineWidth', 1.6);
        else
            hl(k) = plot(ax, x, mu_k, C.curveStyle{k}, ...
                'Color', C.curveGray(k, :), 'LineWidth', 1.0);
        end
    end
    uistack(hl(5), 'top');
    xlim(ax, [0 max(x)]);
    xlabel(ax, 'Training iteration');
    ylabel(ax, 'Common reward (moving avg.)');
    lg = legend(ax, hl, LEARNED, 'Location', 'southeast', 'Box', 'off');
    if isprop(lg, 'NumColumns'), lg.NumColumns = 2; end
    if isprop(lg, 'ItemTokenSize'), lg.ItemTokenSize = [16 8]; end

    save_journal_fig(fh, 'fig16_training_curves', W, HT, FIGDIR);
catch err
    FAILED = record_failure(FAILED, '16: training curves', err);
end

%% ========================== Summary ====================================
fprintf('\nDone. Figures saved in %s\n', FIGDIR);
if isempty(FAILED)
    fprintf('All 16 figures were created.\n');
else
    fprintf(2, '%d block(s) failed:\n', size(FAILED, 1));
    for k = 1:size(FAILED, 1)
        fprintf(2, '  - %s: %s\n', FAILED{k, 1}, FAILED{k, 2});
    end
end

%% ======================================================================
%% Local functions
%% ======================================================================

function journal_bar(M, methods, labels, cond, metric, scale, xlab, C, S, name, w, h, figdir)
% Horizontal journal-style bar chart. No redundant value labels.

n = numel(methods);
v = zeros(1, n);
for i = 1:n
    v(i) = scale * M(methods{i}, cond, metric);
end

if any(~isfinite(v))
    bad = methods(~isfinite(v));
    error('%s contains missing/non-finite values for: %s', name, strjoin(bad, ', '));
end
if any(v < 0)
    error('%s received negative values; use journal_relative for signed data.', name);
end

fh = new_figure(w, h, S);
ax = axes('Parent', fh);
hold(ax, 'on');
style_axes(ax, C, S, false);

maxv = max(v);
if maxv <= 0
    maxv = 1;
end
xt = nice_ticks_positive(maxv * 1.055, 5);
xlim(ax, [0 xt(end)]);
set(ax, 'XTick', xt, ...
        'YLim', [0.45 n + 0.55], ...
        'YDir', 'reverse', ...
        'YTick', 1:n, ...
        'YTickLabel', labels);

for i = 1:n
    draw_hbar(ax, i, 0, v(i), S.barHeight, bar_color(methods{i}, C));
end

idxM5 = find(strcmp(methods, 'M5'), 1);
if ~isempty(idxM5) && idxM5 < n
    plot(ax, [0 xt(end)], [idxM5 + 0.5 idxM5 + 0.5], '-', ...
        'Color', C.separator, 'LineWidth', 0.45);
end

xlabel(ax, xlab);
apply_compact_insets(ax);
save_journal_fig(fh, name, w, h, figdir);
end

function journal_relative(labels, v, xlab, C, S, name, w, h, figdir)
% Signed horizontal bars. Blue = reduction, muted vermillion = increase.

n = numel(v);
if any(~isfinite(v))
    error('%s contains missing/non-finite relative changes.', name);
end

fh = new_figure(w, h, S);
ax = axes('Parent', fh);
hold(ax, 'on');
style_axes(ax, C, S, false);

span = max(abs(v));
if span == 0
    span = 1;
end
lo = min(0, min(v)) - 0.08 * span;
hi = max(0, max(v)) + 0.08 * span;
xt = nice_ticks_signed(lo, hi, 6);

xlim(ax, [xt(1) xt(end)]);
set(ax, 'XTick', xt, ...
        'YLim', [0.45 n + 0.55], ...
        'YDir', 'reverse', ...
        'YTick', 1:n, ...
        'YTickLabel', labels);

for i = 1:n
    if v(i) <= 0
        c = C.negative;
    else
        c = C.positive;
    end
    draw_hbar(ax, i, 0, v(i), S.barHeight, c);
end

plot(ax, [0 0], [0.45 n + 0.55], '-', ...
    'Color', C.zero, 'LineWidth', 0.75);

xlabel(ax, xlab);
apply_compact_insets(ax);
save_journal_fig(fh, name, w, h, figdir);
end

function fh = new_figure(w, h, S)
fh = figure('Color', 'w', ...
            'Visible', S.visible, ...
            'Units', 'inches', ...
            'Position', [1 1 w h], ...
            'Renderer', 'painters', ...
            'InvertHardcopy', 'off');
end

function style_axes(ax, C, S, bothGrid)
set(ax, ...
    'Color', 'w', ...
    'FontName', S.fontName, ...
    'FontSize', S.fontSize, ...
    'LineWidth', S.axisLW, ...
    'Box', 'off', ...
    'TickDir', 'out', ...
    'TickLength', [0.012 0.012], ...
    'XColor', C.axis, ...
    'YColor', C.axis, ...
    'GridColor', C.grid, ...
    'GridAlpha', 1.0, ...
    'GridLineStyle', ':', ...
    'MinorGridLineStyle', ':', ...
    'XMinorGrid', 'off', ...
    'YMinorGrid', 'off');

if bothGrid
    set(ax, 'XGrid', 'on', 'YGrid', 'on');
else
    set(ax, 'XGrid', 'on', 'YGrid', 'off');
end

ax.Layer = 'bottom';
end

function draw_hbar(ax, y, x0, x1, width, color)
yv = y + width / 2 * [-1 -1 1 1];
xv = [x0 x1 x1 x0];
patch(ax, xv, yv, color, ...
    'EdgeColor', 'none', ...
    'FaceAlpha', 1.0);
end

function c = bar_color(method, C)
if strcmp(method, 'M5')
    c = C.accent;
elseif any(strcmp(method, {'M1', 'M2', 'M3', 'M4'}))
    c = C.learn;
else
    c = C.rule;
end
end

function t = nice_ticks_positive(xmax, targetTicks)
if ~isfinite(xmax) || xmax <= 0
    t = [0 1];
    return;
end
step = nice_step(xmax / targetTicks);
hi = ceil(xmax / step) * step;
t = 0:step:hi;
if numel(t) < 3
    step = step / 2;
    t = 0:step:hi;
end
t(abs(t) < 1e-12) = 0;
end

function t = nice_ticks_signed(lo, hi, targetTicks)
if ~isfinite(lo) || ~isfinite(hi) || hi <= lo
    t = [-1 0 1];
    return;
end
step = nice_step((hi - lo) / targetTicks);
lo2 = floor(lo / step) * step;
hi2 = ceil(hi / step) * step;
t = lo2:step:hi2;

if ~any(abs(t) < 1e-10)
    t = sort(unique([t 0]));
end
t(abs(t) < 1e-12) = 0;
end

function s = nice_step(raw)
if raw <= 0 || ~isfinite(raw)
    s = 1;
    return;
end
p = 10 ^ floor(log10(raw));
f = raw / p;

if f <= 1
    nf = 1;
elseif f <= 2
    nf = 2;
elseif f <= 2.5
    nf = 2.5;
elseif f <= 5
    nf = 5;
else
    nf = 10;
end
s = nf * p;
end

function [lo, hi] = padded_limits(v, frac)
vmin = min(v);
vmax = max(v);
span = vmax - vmin;
if span <= 0
    span = max(abs(vmax), 1);
end
lo = vmin - frac * span;
hi = vmax + frac * span;
end

function p = pct_change(newValue, refValue, metricName)
if ~isfinite(newValue) || ~isfinite(refValue)
    error('Non-finite value while computing relative change for %s.', metricName);
end
if refValue == 0
    error('Zero reference value while computing relative change for %s.', metricName);
end
p = 100 * (newValue / refValue - 1);
end

function v = unit_values(R, label, cond, metric, SEEDS, EVALS)
% Learned policies: one value per training seed, averaged over eval seeds.
% Rule-based policies: one value per evaluation seed.

sel = R(strcmp({R.label}, label) & strcmp({R.cond}, cond));
v = [];
if isempty(sel)
    return;
end

if ~isfield(sel(1).m, metric)
    error('Metric "%s" is absent for method "%s".', metric, label);
end

if all([sel.ts] == 0)
    for e = EVALS
        r = sel([sel.es] == e);
        if ~isempty(r)
            v(end + 1) = r(1).m.(metric); %#ok<AGROW>
        end
    end
else
    for s = SEEDS
        r = sel([sel.ts] == s);
        if isempty(r)
            continue;
        end
        v(end + 1) = mean(arrayfun(@(q) q.m.(metric), r)); %#ok<AGROW>
    end
end
end

function f = training_csv(ROOT, model, seed)
% Seed 7 checkpoints live in v3_convergence_anneal_2000, seeds 8-11 in v3_anneal_2000_multiseed.
if seed == 7
    if strcmp(model, 'M5')
        name = 'm5_dual_only_seed7';
    else
        name = [lower(model) '_seed7'];
    end
    f = fullfile(ROOT, 'outputs', 'v3_convergence_anneal_2000', name, 'training_metrics.csv');
else
    f = fullfile(ROOT, 'outputs', 'v3_anneal_2000_multiseed', ...
                 sprintf('%s_seed%d', lower(model), seed), 'training_metrics.csv');
end
end

function [hdr, D] = read_numeric_csv(f)
% Header names + numeric body; works in any MATLAB version and with CRLF files.
fid = fopen(f, 'r');
if fid < 0
    error('Cannot open %s', f);
end
line1 = fgetl(fid);
fclose(fid);
line1 = regexprep(line1, '^\xEF\xBB\xBF', '');          % strip UTF-8 BOM
hdr = strtrim(strsplit(strtrim(line1), ','));
if exist('readmatrix', 'file') == 2 || exist('readmatrix', 'builtin') == 5
    D = readmatrix(f, 'NumHeaderLines', 1);
else
    D = dlmread(f, ',', 1, 0);
end
if size(D, 2) < numel(hdr)
    error('Column mismatch in %s (%d header names, %d columns)', f, numel(hdr), size(D, 2));
end
end

function y = moving_average(x, w)
% Trailing moving average with a growing window at the start.
x = x(:);
c = cumsum(x);
y = zeros(size(x));
for i = 1:numel(x)
    lo = max(1, i - w + 1);
    if lo == 1
        y(i) = c(i) / i;
    else
        y(i) = (c(i) - c(lo - 1)) / (i - lo + 1);
    end
end
end

function apply_compact_insets(ax)
drawnow;
ti = get(ax, 'TightInset');
minInset = [0.02 0.02 0.02 0.02];
set(ax, 'LooseInset', max(ti, minInset));
end

function save_journal_fig(fh, name, w, h, figdir)
set(fh, 'Units', 'inches');
set(fh, 'Position', [1 1 w h]);
set(fh, 'PaperUnits', 'inches', ...
        'PaperSize', [w h], ...
        'PaperPosition', [0 0 w h], ...
        'PaperPositionMode', 'manual');

drawnow;

pngfile = fullfile(figdir, 'png', [name '.png']);
pdffile = fullfile(figdir, 'pdf', [name '.pdf']);

if ~isempty(which('exportgraphics'))
    exportgraphics(fh, pngfile, 'Resolution', 600, 'BackgroundColor', 'white');
    exportgraphics(fh, pdffile, 'ContentType', 'vector', 'BackgroundColor', 'white');
else
    print(fh, pngfile, '-dpng', '-r600');
    print(fh, pdffile, '-dpdf');
end

close(fh);
fprintf('  saved %s\n', name);
end

function FAILED = record_failure(FAILED, blockName, err)
FAILED(end + 1, :) = {blockName, err.message}; %#ok<AGROW>
fprintf(2, '  FAILED [%s]: %s\n', blockName, err.message);
close all;
end