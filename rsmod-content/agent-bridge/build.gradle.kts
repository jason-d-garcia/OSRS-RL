plugins {
    id("base-conventions")
}

dependencies {
    implementation(libs.fastutil)
    implementation(libs.jackson.databind)
    implementation(libs.rsprot.api)
    implementation(projects.api.account)
    implementation(projects.api.pluginCommons)
    implementation(projects.api.pwHash)
    implementation(projects.api.realm)
    implementation(projects.api.registry)
    implementation(projects.content.other.gemstoneCrab)
}
