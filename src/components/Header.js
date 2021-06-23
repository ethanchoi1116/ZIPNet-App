import React, { Component } from "react";
import { Link } from "react-router-dom";
import "./Header.css";

class Header extends Component {
  render() {
    return (
      <header className="d-flex p-5 justify-content-center">
        <Link to={"/"} className="title-link">
          <h1>
            <span>
              <i className="fa fa-user-circle"></i>
            </span>
            &nbsp;
            <span>ZIPNet</span>
          </h1>
        </Link>
      </header>
    );
  }
}

export default Header;
