# Beer-Type-Prediction
Presentation at: https://www.youtube.com/watch?v=HazZTgU0RO8&t=26s

Project Overview:
We built a Bi-Directional LSTM model to classify beers by type (i.e. IPA, Imperial Stout, etc.) based on reviews in the RateBeer dataset. After achieving > 44% accuracy with this model of 89 different categories of beer, we used First Derivative Saliency to make our model interpretable. By doing this, we were able to determine which words were most heavily weighted during classification through our saliency heatmaps. For example, for a fruit beer, the words "lemons" and "candy" were weighted most heavily. We believe other applications of this model could be in attempting to determine tasting notes in various wines or in a similar manner with restaurant reviews.

Please read paper "Classifying Beer Styles Using an Interpretable NLP Model" included for in depth discussion of project
