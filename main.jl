using SpeciesDistributionToolkit
using CairoMakie
using EvoTrees

# Polygon
CHE = SpeciesDistributionToolkit.gadm("CHE")

# Data
provider = RasterData(CHELSA2, BioClim)

# Limits
bbox = (left=5.0, right=12.0, bottom=45.0, top=50.0)

# Trim
layer = [trim(SDMLayer(provider; layer=l, bbox...), CHE) for l in ["BIO2", "BIO5", "BIO14"]]

# GBIF
ouzel = taxon("Turdus torquatus")
gbif_query = ["occurrenceStatus" => "PRESENT", "limit" => 300]
presences = occurrences(ouzel, layer[1], gbif_query... )
while length(presences) < count(presences)
    occurrences!(presences)
end

# Get some pseudo-absences
presencelayer = mask(layer[1], presences)
background = pseudoabsencemask(DistanceToEvent, presencelayer)
bgpoints = backgroundpoints(background, sum(presencelayer))

heatmap(
    layer[2];
    colormap = :deep,
    axis = (; aspect = DataAspect()),
    figure = (; size = (800, 500)),
)
scatter!(mask(presences, CHE); color = :black)
scatter!(bgpoints; color = :red, markersize = 4)
current_figure()

# Make a BRT
nodata!(bgpoints, false)
nodata!(presencelayer, false)
ks = [keys(presencelayer)..., keys(bgpoints)...]
X = Float32.([layer[i][k] for k in ks, i in eachindex(layer)])
y = [ones(Bool, sum(presencelayer))..., zeros(Bool, sum(bgpoints))...]

config = EvoTreeMLE(max_depth=5, nbins=32, nrounds=100)
nobs, nfeats = size(X)
model = fit_evotree(config; x_train=X, y_train=y)
preds = EvoTrees.predict(model, X)

Xp = Float32.([layer[i][k] for k in keys(layer[1]), i in eachindex(layer)])
preds = EvoTrees.predict(model, Xp)

pr = similar(layer[1], Float64)
pr.grid[findall(pr.indices)] .= preds[:,1]

# Plot
f = Figure(; size=(800, 400))
ax = Axis(f[1,1]; aspect=DataAspect())
hm = heatmap!(ax, pr, colormap=:lapaz, colorrange=(0, 1))
Colorbar(f[1,2], hm)
hidedecorations!(ax)
hidespines!(ax)
current_figure()