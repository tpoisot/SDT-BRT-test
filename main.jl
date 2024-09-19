using SpeciesDistributionToolkit
using CairoMakie
using EvoTrees
using Random
using Statistics
using Dates
CairoMakie.activate!(; px_per_unit=3)

# Polygon
CHE = SpeciesDistributionToolkit.gadm("CHE")

# Data
provider = RasterData(WorldClim2, BioClim)

# Limits
bbox = (left=5.0, right=12.0, bottom=45.0, top=50.0, resolution=0.5)

# Trim
layer = [trim(SDMLayer(provider; layer=l, bbox...), CHE) for l in ["BIO1", "BIO2", "BIO5", "BIO14"]]

# GBIF
ouzel = taxon("Turdus torquatus")
gbif_query = ["occurrenceStatus" => "PRESENT", "limit" => 300]
presences = occurrences(ouzel, layer[1], gbif_query...)
while length(presences) < count(presences)
    occurrences!(presences)
end

# Get some pseudo-absences
presencelayer = mask(layer[1], presences)
background = pseudoabsencemask(DistanceToEvent, presencelayer)
bgpoints = backgroundpoints(background, sum(presencelayer))

heatmap(
    layer[2];
    colormap=:deep,
    axis=(; aspect=DataAspect()),
    figure=(; size=(800, 500)),
)
scatter!(mask(presences, CHE); color=:black)
scatter!(bgpoints; color=:red, markersize=4)
current_figure()

# Prepare data to go in a BRT
nodata!(bgpoints, false)
nodata!(presencelayer, false)
ks = [keys(presencelayer)..., keys(bgpoints)...]
X = Float32.([layer[i][k] for k in ks, i in eachindex(layer)])
y = [ones(Bool, sum(presencelayer))..., zeros(Bool, sum(bgpoints))...]

# Shuffle (important)
idx = shuffle(eachindex(y))
X, y = X[idx, :,], y[idx]

# k-folds
K = 10
folds = []
for k in 1:K
    fold_validation = floor.(Int64, LinRange(k, length(y) + k, floor(Int64, length(y) / K)))
    filter!(x -> x <= length(y), fold_validation)
    fold_train = setdiff(eachindex(y), fold_validation)
    push!(folds, (fold_train, fold_validation))
end
folds

#
include("confusion.jl")

config = EvoTreeMLE(max_depth=6, nbins=32, eta=0.05, nrounds=120, L2=0.1, loss=:gaussian_mle)
nobs, nfeats = size(X)
T = LinRange(0.0, 1.0, 250)
M = zeros(ConfusionMatrix, length(T), K)
for k in 1:K
    model = fit_evotree(config; x_train=X[folds[k][1], :], y_train=y[folds[k][1]])
    preds = EvoTrees.predict(model, X[folds[k][2], :])
    for i in eachindex(T)
        M[i, k] = ConfusionMatrix(preds[:, 1], y[folds[k][2]], Float32(T[i]))
    end
end

# tuning curve
f = Figure()
ax = Axis(f[1, 1])
lines!(ax, T, vec(mean(mcc.(M); dims=2)))
Tmax = T[last(findmax(vec(mean(mcc.(M); dims=2))))]
current_figure()

# ROC
f = Figure()
ax = Axis(f[1, 1]; aspect=1)
for i in 1:K
    lines!(ax, fpr.(M[:, i]), tpr.(M[:, i]), color=:black)
end
current_figure()

[auc(fpr.(M[:, k]), tpr.(M[:, k])) for k in 1:K]

# PR
f = Figure()
ax = Axis(f[1, 1]; aspect=1)
for i in 1:K
    lines!(ax, tpr.(M[:, i]), ppv.(M[:, i]), color=:black)
end
current_figure()

[auc(tpr.(M[:, k]), ppv.(M[:, k])) for k in 1:K]

Xp = Float32.([layer[i][k] for k in keys(layer[1]), i in eachindex(layer)])
model = fit_evotree(config; x_train=X, y_train=y)
preds = EvoTrees.predict(model, Xp)

pr = similar(layer[1], Float64)
pr.grid[findall(pr.indices)] .= preds[:, 1]

unc = similar(layer[1], Float64)
unc.grid[findall(unc.indices)] .= preds[:, 2]

# Plot
f = Figure(; size=(800, 600))
ax_current = Axis(f[1, 2]; aspect=DataAspect(), title="P(presence)")
hm = heatmap!(ax_current, pr, colormap=:lipari, colorrange=(0, 1))
Colorbar(f[1, 1], hm)
ax_current_range = Axis(f[2, 2]; aspect=DataAspect())
heatmap!(ax_current_range, pr .>= Tmax, colormap=[:lightgrey, :green])
ax_unc = Axis(f[1, 3]; aspect=DataAspect(), title="Standard deviation")
hm2 = heatmap!(ax_unc, unc, colormap=[colorant"#000000ff", colorant"#ffffff00"])
Colorbar(f[1, 4], hm2)
for ax in filter(c -> c isa Axis, f.content)
    hidedecorations!(ax, label=false)
    hidespines!(ax)
end
current_figure()

# Partial responses
f = Figure(; size=(900, 600))
ax_maps = [Axis(f[i, 1]; aspect=DataAspect()) for i in 1:3]
ax_maps_2 = [Axis(f[i, 3]; aspect=DataAspect()) for i in 1:3]
ax_resp = [Axis(f[i, 2]) for i in 1:3]
for i in 1:3
    mXp = copy(Xp)
    for j in 1:3
        if j != i
            mXp[:, j] .= mean(X[:, j])
        end
    end
    pr1 = similar(layer[1], Float64)
    pr1.grid[findall(pr1.indices)] .= EvoTrees.predict(model, mXp)[:, 1]
    heatmap!(ax_maps[i], pr1, colormap=:navia, colorrange=(0, 1))
    heatmap!(ax_maps_2[i], (pr1-pr)/(pr1+pr), colormap=:roma, colorrange=(-0.2, 0.2))

    xx = LinRange(extrema(layer[i])..., 300)
    prX = zeros(Float64, length(xx), 3)
    for j in 1:3
        if j != i
            prX[:, j] .= mean(X[:, j])
        else
            prX[:, j] .= xx
        end
    end
    prpr = EvoTrees.predict(model, prX)
    band!(ax_resp[i], xx, prpr[:, 1] .- prpr[:, 2] ./ 2, prpr[:, 1] .+ prpr[:, 2] ./ 2, color=:grey, alpha=0.3)
    lines!(ax_resp[i], xx, prpr[:, 1], color=:black)
end
ylims!.(ax_resp, 0.0, 1.0)
hidedecorations!.(ax_maps)
hidespines!.(ax_maps)
hidedecorations!.(ax_maps_2)
hidespines!.(ax_maps_2)
current_figure()
