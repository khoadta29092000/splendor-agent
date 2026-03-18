FROM node:20-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

RUN npm run build

# KHÔNG hardcode 4000 nữa
ENV PORT=10000

EXPOSE 10000

CMD ["node", "dist/index.js"]