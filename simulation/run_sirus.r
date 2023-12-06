library(sirus)
library(argparse)


parser <- ArgumentParser()
parser$add_argument("--input",dest="input_file", nargs=1, help="Input file with subgroup data")
parser$add_argument("--output",dest="output_file", nargs=1, help="Output file for sirus rules")
parser$add_argument("--depth", dest="depth", nargs=1, default=2, help="Max number of literals in rule", type="integer")
parser$add_argument("--cv", action="store_true", default=FALSE, help="Run sirus.cv to determine p0")

args <- parser$parse_args()
print(args$depth)


sirus_df <- read.csv(args$input_file)


# last column is worst-performing subset membership indicator
subset_membership <- sirus_df[, ncol(sirus_df)]
# predict subset membership from X (subgroup defining features)
sirus_X <- sirus_df[, -c(ncol(sirus_df))]

if(args$cv){
    scv <- sirus.cv(sirus_X, subset_membership, max.depth=args$depth)
    print("p0=")
    print(scv$p0.pred)
    sf <- sirus.fit(sirus_X, subset_membership, p0=scv$p0.pred, max.depth=args$depth)
} else {
    sf <- sirus.fit(sirus_X, subset_membership, max.depth=args$depth)
}

cat(sirus.print(sf), file=args$output_file, sep="\n")