get_theme_palette <- function() {
    
    ggthemr::define_palette(
        swatch = c("#000000",
                   "#377EB8", "#E41A1C", "#4DAF4A", "#984EA3",
                   "#FF7F00", "#FFFF33", "#A65628", "#F781BF"),
        gradient = c(lower = "#377EB8", upper = "#E41A1C")
    )
    
}


theme_our <- function(base_size = 13) {
    
    theme_bw(base_size, base_family = "Canaro Medium") +
        theme(legend.position = "bottom")
    
}


update_font_defaults <- function() {
    
    update_geom_defaults("text", list(family = "Canaro Medium"))
    update_geom_defaults("label", list(family = "Canaro Medium"))
    
}

set_our_theme <- function(base_size = 13)  {
    
    ggthemr::ggthemr(get_theme_palette())
    
    theme_set(theme_our(base_size = base_size))
    
    update_font_defaults()
    
}