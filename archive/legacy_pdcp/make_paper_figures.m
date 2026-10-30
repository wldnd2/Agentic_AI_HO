%% make_paper_figures.m  --  plain journal-style figures (grouped bars per metric)
% Layout follows common IEEE practice: one panel per metric, the two
% evaluation conditions on the x-axis, one bar per method, one shared legend
% on top, no numbers printed inside the plots (values are given in the text
% and tables).
%
% Output (ROOT/fig/pdf and ROOT/fig/png):
%   fig01_training      training convergence (common reward)
%   fig02_comparison    all baselines + MADQN + MAPPO (M1) + M5, six metrics
%   fig03_ablation      capability ladder M1-M5 and M1D, four metrics
%   fig04_behavior      how CHO preparations are used, three metrics
%   fig05_tradeoff      HO executions vs outage of the learned variants
%
% Data: ROOT/outputs/unified_eval/raw/*.json (copy the M1D_*.json files there),
%       training logs under ROOT/outputs/...
clear; close all;

%% ------------------------------ settings ---------------------------------
ROOT = '/Users/jeonjiwoong/Desktop/CHO_simul';
K.font      = 'Times New Roman';
K.fs        = 9;            % font size
K.lw        = 0.5;          % bar edge width
K.showSD    = false;        % true: draw error bars (standard deviation across seeds)
K.colW      = 3.5;          % single column width (in)
K.pageW     = 7.16;         % double column width (in)

% method sets ----------------------------------------------------------------
CMP.keys  = {'CHO-A3-TTT', 'CHO-A3-TTT+Dup', 'Max-Observed-SINR', 'Always-Dup', ...
             'Threshold-Dup', 'MADQN', 'M1', 'M5'};
CMP.names = {'CHO', 'CHO+Dup', 'Max-SINR', 'Always-Dup', 'Threshold-Dup', ...
             'MADQN', 'MAPPO (M1)', 'M5 (proposed)'};
CMP.col   = [0.000 0.447 0.741;  0.301 0.745 0.933;  0.929 0.694 0.125;
             0.494 0.184 0.556;  0.466 0.674 0.188;  0.250 0.250 0.250;
             0.600 0.600 0.600;  0.850 0.325 0.098];
% methods shown in the training-curve figure (missing logs are skipped)
TRN.keys  = {'MADQN', 'M1', 'M2', 'M3', 'M4', 'M5', 'M1D'};
TRN.names = {'MADQN', 'MAPPO (M1)', 'M2', 'M3', 'M4', 'M5', 'M1D'};
TRN.col   = [0.250 0.250 0.250;  0.600 0.600 0.600;  0.301 0.745 0.933;
             0.466 0.674 0.188;  0.929 0.694 0.125;  0.850 0.325 0.098;
             0.000 0.447 0.741];
ABL.keys  = {'M1', 'M2', 'M3', 'M4', 'M5', 'M1D'};
ABL.names = {'M1', 'M2', 'M3', 'M4', 'M5', 'M1D'};
ABL.col   = [0.600 0.600 0.600;  0.301 0.745 0.933;  0.466 0.674 0.188;
             0.929 0.694 0.125;  0.850 0.325 0.098;  0.000 0.447 0.741];
CONDS  = {'nominal', 'stress'};
CLABEL = {'Nominal', 'Compound stress'};
SEEDS  = [7 8 9 10 11];
LAMBDA = 0.25;

RAW = fullfile(ROOT, 'outputs', 'unified_eval', 'raw');
OUT = fullfile(ROOT, 'fig');
if ~exist(fullfile(OUT, 'pdf'), 'dir'), mkdir(fullfile(OUT, 'pdf')); end
if ~exist(fullfile(OUT, 'png'), 'dir'), mkdir(fullfile(OUT, 'png')); end
set(groot, 'DefaultAxesFontName', K.font, 'DefaultTextFontName', K.font, ...
    'DefaultAxesFontSize', K.fs, 'DefaultTextFontSize', K.fs);

%% ------------------------------ load data --------------------------------
files = dir(fullfile(RAW, '*.json'));
if isempty(files), error('No result files in %s', RAW); end
R = struct('label', {}, 'cond', {}, 'ts', {}, 'es', {}, 'm', {});
for k = 1:numel(files)
    j = jsondecode(fileread(fullfile(files(k).folder, files(k).name)));
    if isfield(j, 'ablation') && ~strcmp(j.ablation, 'none'), continue; end
    m = j.metrics;
    m.delivery_ratio = 100 * m.delivered_packets / m.arrivals;
    m.deadline_pct   = 100 * m.deadline_violation_rate;
    m.commit_pct     = 100 * m.prepare_to_commit_rate;
    m.reject_pct     = 100 * m.preparation_rejection_rate;
    m.common_reward  = -m.service_deficit - LAMBDA * m.resource_cost;
    R(end + 1) = struct('label', j.model, 'cond', j.condition, 'ts', j.train_seed, ...
                        'es', j.eval_seed, 'm', m); %#ok<SAGROW>
end
fprintf('%d result files loaded\n', numel(R));
FAILED = {};

%% ------------------------- fig01: training curves ------------------------
try
    fh = newfig(K.colW, 2.0); ax = newax(fh, K);
    mk = {'x', 'o', 's', '^', 'd', 'p', 'v'};
    hl = []; names = {};
    for k = 1:numel(TRN.keys)
        [x, y] = training_curve(ROOT, TRN.keys{k}, SEEDS, LAMBDA);
        if isempty(y), continue; end
        idx = 1:200:numel(x);
        hl(end + 1) = plot(ax, x, y, '-', 'Color', TRN.col(k, :), 'LineWidth', 1.0); %#ok<SAGROW>
        plot(ax, x(idx), y(idx), mk{k}, 'Color', TRN.col(k, :), 'MarkerSize', 3.5, ...
             'MarkerFaceColor', 'w', 'HandleVisibility', 'off');
        names{end + 1} = TRN.names{k}; %#ok<SAGROW>
    end
    xlabel(ax, 'Training iteration'); ylabel(ax, 'Common reward');
    xlim(ax, [0 2000]); ylim(ax, [-0.24 -0.04]);   % room for the legend below the curves
    lg = legend(ax, hl, names, 'Location', 'southeast');
    if isprop(lg, 'NumColumns'), lg.NumColumns = 2; end
    savefig_all(fh, 'fig01_training', K.colW, 2.0, OUT);
catch err
    FAILED = fail(FAILED, 'fig01_training', err);
end

%% ------------------------- fig02: comparison -----------------------------
try
    P = { 'outage_ratio',     1, '(a) Outage ratio';
          'service_deficit',  1, '(b) Service deficit';
          'delivery_ratio',   1, '(c) Packet delivery ratio (%)';
          'deadline_pct',     1, '(d) Deadline-violation rate (%)';
          'pingpong_rate',    1, '(e) Ping-pong ratio';
          'handover_failure', 1, '(f) Failed handovers per epoch'};
    panel_figure(R, CMP, P, CONDS, CLABEL, SEEDS, K, 2, 3, K.pageW, 3.1, ...
                 'fig02_comparison', OUT);
catch err
    FAILED = fail(FAILED, 'fig02_comparison', err);
end

%% ------------------------- fig03: capability ablation --------------------
try
    P = { 'outage_ratio',     1, '(a) Outage ratio';
          'handover_started', 1, '(b) HO executions per epoch';
          'pingpong_rate',    1, '(c) Ping-pong ratio';
          'common_reward',    1, '(d) Common reward'};
    panel_figure(R, ABL, P, CONDS, {'Nominal', 'Stress'}, SEEDS, K, 1, 4, K.pageW, 1.85, ...
                 'fig03_ablation', OUT);
catch err
    FAILED = fail(FAILED, 'fig03_ablation', err);
end

%% ------------------------- fig04: behavior -------------------------------
try
    P = { 'commit_pct',             1, '(a) Preparations ending in a switch (%)';
          'reject_pct',             1, '(b) Rejected preparation requests (%)';
          'mean_prepared_duration', 1, '(c) Preparation holding time (epochs)'};
    panel_figure(R, CMP, P, CONDS, CLABEL, SEEDS, K, 1, 3, K.pageW, 1.8, ...
                 'fig04_behavior', OUT);
catch err
    FAILED = fail(FAILED, 'fig04_behavior', err);
end

%% ------------------------- fig05: trade-off ------------------------------
try
    fh = newfig(K.colW, 2.4); ax = newax(fh, K);
    mk = {'o', 's', '^', 'd', 'p', 'v'};
    hs = []; names = {}; xs = []; ys = [];
    for k = 1:numel(ABL.keys)
        if ~any(strcmp({R.label}, ABL.keys{k})), continue; end
        xv = mean(unit_values(R, ABL.keys{k}, 'nominal', 'handover_started', SEEDS));
        yv = mean(unit_values(R, ABL.keys{k}, 'nominal', 'outage_ratio', SEEDS));
        xs(end + 1) = xv; ys(end + 1) = yv; %#ok<SAGROW>
        hs(end + 1) = plot(ax, xv, yv, mk{k}, 'MarkerSize', 7, 'LineWidth', 0.8, ...
            'MarkerEdgeColor', 'k', 'MarkerFaceColor', ABL.col(k, :)); %#ok<SAGROW>
        names{end + 1} = ABL.names{k}; %#ok<SAGROW>
    end
    xlabel(ax, 'Handover executions per epoch'); ylabel(ax, 'Outage ratio');
    lg = legend(ax, hs, names, 'Location', 'northwest');
    if isprop(lg, 'NumColumns'), lg.NumColumns = 3; end
    dx = max(xs) - min(xs); dy = max(ys) - min(ys);
    xlim(ax, [min(xs) - 0.25 * dx, max(xs) + 0.25 * dx]);
    ylim(ax, [min(ys) - 0.3 * dy, max(ys) + 0.9 * dy]);
    savefig_all(fh, 'fig05_tradeoff', K.colW, 2.4, OUT);
catch err
    FAILED = fail(FAILED, 'fig05_tradeoff', err);
end

fprintf('\nFigures written to %s\n', OUT);
if isempty(FAILED)
    fprintf('All figures created.\n');
else
    for k = 1:size(FAILED, 1), fprintf(2, 'FAILED %s: %s\n', FAILED{k, 1}, FAILED{k, 2}); end
end

%% =========================================================================
%% local functions
%% =========================================================================
function panel_figure(R, S, P, conds, clabel, seeds, K, nr, nc, w, h, name, out)
% one panel per metric; x-axis = condition; one bar per method; shared legend
fh = newfig(w, h);
nm = numel(S.keys);
present = cellfun(@(k) any(strcmp({R.label}, k)), S.keys);
keys = S.keys(present); names = S.names(present); col = S.col(present, :);
nm = numel(keys);
ncol = legend_columns(names, w, K);           % as many columns as fit in one row
nrow = ceil(nm / ncol); ncol = ceil(nm / nrow); % balance the rows (e.g. 4 + 4)
legH   = (0.20 + 0.17 * nrow) / h;              % legend band (normalized)
titleH = 0.22 / h; tickH = 0.24 / h;            % title above, tick labels below
left = 0.42 / w; right = 0.06 / w; gapX = 0.42 / w;
pw = (1 - left - right - (nc - 1) * gapX) / nc;
ph = (1 - legH - nr * (titleH + tickH)) / nr;
hb = [];
for p = 1:size(P, 1)
    r = ceil(p / nc); c = p - (r - 1) * nc;
    pos = [left + (c - 1) * (pw + gapX), 1 - legH - titleH - ph - (r - 1) * (ph + titleH + tickH), pw, ph];
    ax = axes('Parent', fh, 'Position', pos); hold(ax, 'on'); box(ax, 'on');
    Y = zeros(numel(conds), nm); E = Y;
    for i = 1:numel(conds)
        for k = 1:nm
            v = P{p, 2} * unit_values(R, keys{k}, conds{i}, P{p, 1}, seeds);
            Y(i, k) = mean(v);
            if numel(v) > 1, E(i, k) = std(v); end
        end
    end
    b = bar(ax, 1:numel(conds), Y, 0.85);
    for k = 1:nm
        set(b(k), 'FaceColor', col(k, :), 'EdgeColor', 'k', 'LineWidth', K.lw);
    end
    if K.showSD
        for k = 1:nm
            xk = (1:numel(conds)) + (k - (nm + 1) / 2) * 0.85 / nm;
            errorbar(ax, xk, Y(:, k), E(:, k), 'k.', 'LineWidth', 0.5, 'MarkerSize', 1);
        end
    end
    set(ax, 'XTick', 1:numel(conds), 'XTickLabel', clabel, 'XLim', [0.5 numel(conds) + 0.5], ...
        'FontName', K.font, 'FontSize', K.fs, 'LineWidth', 0.5, 'TickDir', 'in', ...
        'YGrid', 'on', 'GridLineStyle', ':');
    lo = min(0, min(Y(:))); hi = max(Y(:));
    if lo < 0, ylim(ax, [1.15 * lo, 0]); else, ylim(ax, [0, 1.12 * hi]); end
    title(ax, P{p, 3}, 'FontWeight', 'normal', 'FontSize', K.fs);
    if p == 1, hb = b; end
end
draw_legend(fh, names, col, legH, ncol, w, K);
savefig_all(fh, name, w, h, out);
end

function draw_legend(fh, names, col, legH, ncol, w, K)
% legend drawn by hand in a band at the top of the figure (identical in every
% MATLAB/Octave version): colored boxes with black edges and labels
ax = axes('Parent', fh, 'Position', [0 1 - legH 1 legH], 'Visible', 'off', ...
          'XLim', [0 w], 'YLim', [0 1]);
hold(ax, 'on');
chw = 0.062 * K.fs / 9;                         % approx. character width (in)
box = 0.16; gap = 0.06; sep = 0.22;             % inches
widths = cellfun(@(n) box + gap + chw * numel(n) + sep, names);
nrow = ceil(numel(names) / ncol);
for r = 1:nrow
    idx = (r - 1) * ncol + 1 : min(r * ncol, numel(names));
    x = (w - sum(widths(idx)) + sep) / 2;
    y = 1 - (r - 0.5) / nrow;
    for k = idx
        patch(ax, x + [0 box box 0], y + 0.18 / nrow * [-1 -1 1 1] * 1.1, col(k, :), ...
              'EdgeColor', 'k', 'LineWidth', K.lw);
        text(ax, x + box + gap, y, names{k}, 'FontName', K.font, 'FontSize', K.fs, ...
             'VerticalAlignment', 'middle');
        x = x + widths(k);
    end
end
end

function ncol = legend_columns(names, w, K)
% largest number of legend columns whose rows fit into the figure width
chw = 0.062 * K.fs / 9; box = 0.16; gap = 0.06; sep = 0.22;
widths = cellfun(@(n) box + gap + chw * numel(n) + sep, names);
nm = numel(names);
for ncol = nm:-1:1
    ok = true;
    for r = 1:ceil(nm / ncol)
        idx = (r - 1) * ncol + 1 : min(r * ncol, nm);
        if sum(widths(idx)) - sep > w - 0.15, ok = false; break; end
    end
    if ok, return; end
end
ncol = 1;
end

function t = tinv975(df)
T = [12.706 4.303 3.182 2.776 2.571 2.447 2.365 2.306 2.262 2.228];
if df <= numel(T), t = T(df); else, t = 1.96; end
end

function v = unit_values(R, label, cond, metric, seeds)
% learned: one value per training seed (mean over evaluation seeds);
% rule-based (train seed 0): one value per evaluation seed
sel = R(strcmp({R.label}, label) & strcmp({R.cond}, cond));
v = [];
if isempty(sel), return; end
if all([sel.ts] == 0)
    v = arrayfun(@(q) q.m.(metric), sel);
else
    for s = seeds
        r = sel([sel.ts] == s);
        if ~isempty(r), v(end + 1) = mean(arrayfun(@(q) q.m.(metric), r)); end %#ok<AGROW>
    end
end
end

function [x, y] = training_curve(ROOT, model, seeds, lambda)
Y = []; x = [];
for s = seeds
    if s == 7 && ~any(strcmp(model, {'M1D', 'MADQN'}))
        d = lower(model); if strcmp(model, 'M5'), d = 'm5_dual_only'; end
        f = fullfile(ROOT, 'outputs', 'v3_convergence_anneal_2000', [d '_seed7'], 'training_metrics.csv');
    else
        f = fullfile(ROOT, 'outputs', 'v3_anneal_2000_multiseed', sprintf('%s_seed%d', lower(model), s), ...
                     'training_metrics.csv');
    end
    if ~exist(f, 'file'), continue; end
    fid = fopen(f, 'r'); hdr = strtrim(strsplit(strtrim(fgetl(fid)), ',')); fclose(fid);
    D = dlmread(f, ',', 1, 0);
    sd = D(:, strcmp(hdr, 'service_deficit')); rc = D(:, strcmp(hdr, 'resource_cost'));
    it = D(:, strcmp(hdr, 'iteration'));
    if isempty(sd) || isempty(rc) || isempty(it), continue; end
    r = movmean_simple(-sd - lambda * rc, 50);
    Y = [Y; r(:)']; x = it(:)'; %#ok<AGROW>
end
if isempty(Y), y = []; else, y = mean(Y, 1); end
end

function y = movmean_simple(x, w)
x = x(:); c = cumsum(x); y = zeros(size(x));
for i = 1:numel(x)
    lo = max(1, i - w + 1);
    if lo == 1, y(i) = c(i) / i; else, y(i) = (c(i) - c(lo - 1)) / (i - lo + 1); end
end
end

function fh = newfig(w, h)
fh = figure('Color', 'w', 'Visible', 'off', 'Units', 'inches', 'Position', [1 1 w h]);
end

function ax = newax(fh, K)
ax = axes('Parent', fh); hold(ax, 'on'); box(ax, 'on');
set(ax, 'FontName', K.font, 'FontSize', K.fs, 'LineWidth', 0.5, 'TickDir', 'in', ...
    'XGrid', 'on', 'YGrid', 'on', 'GridLineStyle', ':');
end

function savefig_all(fh, name, w, h, out)
set(fh, 'Units', 'inches', 'Position', [1 1 w h], 'PaperUnits', 'inches', ...
    'PaperSize', [w h], 'PaperPosition', [0 0 w h]);
drawnow;
pdf = fullfile(out, 'pdf', [name '.pdf']); png = fullfile(out, 'png', [name '.png']);
if ~isempty(which('exportgraphics'))
    exportgraphics(fh, pdf, 'ContentType', 'vector', 'BackgroundColor', 'white');
    exportgraphics(fh, png, 'Resolution', 600, 'BackgroundColor', 'white');
else
    print(fh, pdf, '-dpdf'); print(fh, png, '-dpng', '-r600');
end
close(fh);
fprintf('  saved %s\n', name);
end

function F = fail(F, name, err)
F(end + 1, :) = {name, err.message};
fprintf(2, '  FAILED [%s]: %s\n', name, err.message);
close all;
end