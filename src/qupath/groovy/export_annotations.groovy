/*
 * Export your annotations (not the overlay tiles) for every image. QuPath 0.7.0.
 *
 *   Automate > Script editor > File > Open... (this file)
 *   Run > Run for project (without save) -> move all images to the right -> OK
 *
 * Writes <package>/annotations/<code>.geojson (full-resolution pixel coordinates,
 * with class names). Send back the annotations/ folder and review_form.csv.
 * Save your work (File > Save) before running: batch runs read the saved data.
 */
def name = getProjectEntry().getImageName()
def match = name =~ /^R\d+/
def code = match.find() ? match.group() : name.replaceFirst(/\.[^.]+$/, "")
// The project's own location (project.qpproj or its folder) -> the project folder.
def projectFile = getProject().getPath().toFile()
def projectDir = projectFile.isDirectory() ? projectFile : projectFile.getAbsoluteFile().getParentFile()
def packageDir = projectDir.getAbsoluteFile().getParentFile() ?: projectDir
def outDir = new File(packageDir.getPath(), "annotations")
outDir.mkdirs()

def annotations = getAnnotationObjects()
def out = new File(outDir, code + ".geojson")
exportObjectsToGeoJson(annotations, out.getAbsolutePath(), "FEATURE_COLLECTION")
println "${code}: ${annotations.size()} annotations -> ${out}"
