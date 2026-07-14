
path = ARGS[1]
content = read(path, String)
lines = split(content, "\n")
newLines = []
for line in lines
    # Check if this line is the one we want to remove
    if occursin("output_csv=joinpath(@__DIR__, \"..\", \"outputs\", \"function_tour\", \"nb_compare.csv\")", line) && !occursin("output_dir = joinpath", line)
        # This is the line we want to skip (the duplicate one)
        continue
    end
    push!(newLines, line)
end
newContent = join(newLines, "\n")
open(path, "w") do io
    write(io, newContent)
end
