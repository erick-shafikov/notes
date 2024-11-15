import { Get, Controller, startApp, Injectable, Inject, QueryParameter } from './framework'

@Injectable('CitiesDB')
class CitiesDB {
  public getCities() {
    console.log('CitiesDB - Getting a list of cities')
    return ['London', 'New York', 'Dublin']
  }
}

@Controller('/api')
class WeatherController {
  @Get('/forecast')
  public get(@QueryParameter('cityName') city: string) {
    return {
      apiVersion: 'v1',
      temperature: 20,
      humidity: 80,
      city: city,
    }
  }
}

@Controller('/api')
class CitiesController {
  @Inject('CitiesDB')
  private citiesDb: CitiesDB

  @Get('/cities')
  public get() {
    return {
      cities: this.citiesDb.getCities(),
    }
  }
}

startApp()
