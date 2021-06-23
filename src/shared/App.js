// App.js
import React, { Component } from "react";
import { Route } from "react-router-dom";
import { Home } from "../pages";
import { Result } from "../pages";
import "./App.css";
import Header from "../components/Header";
import Footer from "../components/Footer";

class App extends Component {
  render() {
    return (
      <div className="container-fluid d-flex flex-column justify-content-between text-white">
        <Header></Header>
        <Route exact path="/" component={Home} />
        <Route exact path="/result" component={Result} />
        <Footer></Footer>
      </div>
    );
  }
}

export default App;
