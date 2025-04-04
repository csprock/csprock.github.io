FROM ruby:latest

WORKDIR /site

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy Gemfile and install gems
COPY Gemfile .
RUN bundle install

# Copy the rest of the site
COPY . .

# Expose port 4000 for Jekyll server
EXPOSE 4000

# Command to serve the site
CMD ["bundle", "exec", "jekyll", "serve", "--host", "0.0.0.0"]