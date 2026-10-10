/*
 * Attach the EBV attention overlay to every image in the project. QuPath 0.7.0.
 *
 *   Automate > Script editor > File > Open... (this file)
 *   Run > Run for project -> move all images to the right -> OK
 *   ("Run for project" saves each image afterwards.)
 *
 * Expected layout (the review package):
 *   <package>/overlays/R01.geojson, R02.geojson, ...
 *   <package>/project/      <- the QuPath project, created inside the package
 * Images are matched by name without extension (R01.svs -> R01.geojson).
 * Safe to re-run: earlier overlay tiles are removed first; annotations are kept.
 *
 * Template: build_package.py fills CLASSES from src/qupath/config.py.
 */
import qupath.lib.common.ColorTools
import qupath.lib.objects.classes.PathClass

// Leave empty to find <package>/overlays automatically. If your overlays live
// elsewhere, put the full folder path here, e.g. "/Users/me/EBV_review/overlays".
def OVERLAY_DIR = ""

def CLASSES = {{ANNOTATION_CLASSES_GROOVY}}

println "import_overlays.groovy v3"
def name = getProjectEntry().getImageName()
def match = name =~ /^R\d+/
def code = match.find() ? match.group() : name.replaceFirst(/\.[^.]+$/, "")
// The project's own location (project.qpproj or its folder) -> the project folder.
def projectFile = getProject().getPath().toFile()
def projectDir = projectFile.isDirectory() ? projectFile : projectFile.getAbsoluteFile().getParentFile()
def packageDir = projectDir.getAbsoluteFile().getParentFile()
def candidates = OVERLAY_DIR ? [new File(OVERLAY_DIR)] : [
        packageDir ? new File(packageDir.getPath(), "overlays") : null,
        new File(projectDir.getPath(), "overlays")].findAll { it != null }
def overlayDir = candidates.find { it.isDirectory() }
if (overlayDir == null) {
    println "ERROR ${name}: no overlays folder. Looked in: ${candidates*.absolutePath}. " +
            "Set OVERLAY_DIR at the top of this script."
    return
}
def overlay = new File(overlayDir, code + ".geojson")
if (!overlay.exists()) {
    println "SKIP ${name}: no ${overlay.name} in ${overlayDir}"
    return
}

removeObjects(getTileObjects(), true)
importObjectsFromFile(overlay.getAbsolutePath())
println "${name}: ${getTileObjects().size()} overlay tiles"

// Offer the annotation classes in the class list (keeps any existing classes).
def project = getProject()
def existing = project.getPathClasses()
def wanted = CLASSES.collect { PathClass.fromString(it[0], ColorTools.packRGB(it[1], it[2], it[3])) }
def merged = existing + wanted.findAll { !(it in existing) }
if (project.setPathClasses(merged))
    project.syncChanges()
