FROM node:22-alpine

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# the backend is the compose service name when running in a container
ENV VITE_API_TARGET=http://backend:8000

EXPOSE 5173
CMD ["npm", "run", "dev"]
